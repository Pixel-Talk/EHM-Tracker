import torch
import numpy as np
import cv2,os
import torch.nn.functional as nnfunc
from tqdm.auto import tqdm
from pytorch3d.transforms import matrix_to_rotation_6d,rotation_6d_to_matrix
from pytorch3d.transforms import matrix_to_rotation_6d,rotation_6d_to_matrix
from src.utils.draw import draw_landmarks
from ...modules.flame import FLAME
from ...modules.renderer.head_renderer import Renderer
from ...utils.rprint import rlog as log
from ...utils.graphics import GS_Camera
from ...utils.helper import build_minibatch
from ...losses import Landmark2DLoss
np.random.seed(0)

class FlameOptimizer(object):
    def __init__(self, flame_assets_dir, device='cuda:0', image_size=512,tanfov=1/12, ):
        self.flame_assets_dir = flame_assets_dir
        self.device = device
        self.image_size = image_size
        self.focal_length = 1/tanfov
        self.flame = FLAME(flame_assets_dir).to(self.device)
        self.renderer = Renderer(flame_assets_dir, image_size,focal_length=self.focal_length).to(self.device)
        self.face2d_lmk_distance = Landmark2DLoss(self.flame.lmk_203_left_indices,
                                                      self.flame.lmk_203_right_indices,
                                                      self.flame.lmk_203_front_indices,
                                                      self.flame.lmk_mp_indices)
    
    def build_cameras_kwargs(self, batch_size):
        screen_size = torch.tensor([self.image_size, self.image_size], device=self.device).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(), 
            'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        return cameras_kwargs

    def optimize(self, track_frames, batch_base,id_share_params_result=None, batch_id=0, steps=1001, share_id=True,batch_imgs=None,interval=1):
        batch_size = len(track_frames)
        batch_flame, gt_lmk_203, gt_lmk_fan, gt_lmk_mp = batch_base['flame_coeffs'], batch_base['head_lmk_203'], batch_base['head_lmk_70'], batch_base['head_lmk_mp']
        g_flame_shape=torch.tensor(id_share_params_result['flame_shape'],device=self.device)
        if share_id:
            batch_flame['shape_params'] = g_flame_shape.expand(batch_size, -1)
        
        landmark_lst_dct = dict(lmk_203=gt_lmk_203, lmk_fan=gt_lmk_fan, lmk_mp=gt_lmk_mp)
        
        batch_flame = {k: v.squeeze(1) for k, v in batch_flame.items()}
        batch_flame['expression_params'].requires_grad = True
        batch_flame['shape_params'].requires_grad = True
        batch_flame['eyelid_params'].requires_grad = True
        batch_flame['jaw_params'].requires_grad = True
        batch_flame['pose_params'].requires_grad = True
        batch_flame['camera_RT_params'].requires_grad = True
        
        g_flame_shape.requires_grad=True
        gl_T = batch_flame['camera_RT_params'][:, :3, 3].detach().clone()
        gl_R = batch_flame['camera_RT_params'][:, :3, :3].detach().clone()
        gl_R_6d=matrix_to_rotation_6d(gl_R).detach().clone()
        gl_T.requires_grad=True
        gl_R_6d.requires_grad=True
        opt_p = torch.optim.AdamW([
            {'params': [g_flame_shape], 'lr': 1e-4},
            {'params': [batch_flame['expression_params']], 'lr': 1e-3},
            {'params': [batch_flame['eyelid_params']], 'lr': 2e-4},
            {'params': [batch_flame['jaw_params']], 'lr': 1e-3},
            {'params': [batch_flame['pose_params']], 'lr': 1e-3},
            {'params': [gl_T], 'lr': 1e-5},
            {'params': [gl_R], 'lr': 1e-5},
            {'params': [gl_R_6d], 'lr': 1e-5},
        ])

        cameras_kwargs = self.build_cameras_kwargs(batch_size)
        cameras = GS_Camera(**cameras_kwargs).to(self.device)
        t_bar = tqdm(range(steps), desc='Start tuning FLAME params')
        for i_step in t_bar:
            batch_flame['shape_params']=g_flame_shape.expand(batch_size, -1)
            ret_dict = self.flame(batch_flame)
            lmk_losses = []
            R = rotation_6d_to_matrix(gl_R_6d)
            T = gl_T
            for k in landmark_lst_dct.keys():
                t_lmk = cameras.transform_points_screen(ret_dict[k], R=R, T=T)[..., :2]
                t_w = 2* 5
                if '203' in k: t_w = 10*5
                lmk_losses.append(t_w * self.face2d_lmk_distance(t_lmk, landmark_lst_dct[k], cam=batch_flame['cam']))
            landmark_loss2 = sum(lmk_losses) 

            if share_id:
                temp_loss = torch.mean(torch.square(batch_flame['expression_params'][1:] - batch_flame['expression_params'][:-1])) * 10/interval
                temp_loss += torch.mean(torch.square(batch_flame['shape_params'][1:] - batch_flame['shape_params'][:-1])) * 1e3/interval
                temp_loss += torch.mean(torch.square(batch_flame['jaw_params'][1:] - batch_flame['jaw_params'][:-1])) * 1e4/interval
                temp_loss += torch.mean(torch.square(batch_flame['pose_params'][1:] - batch_flame['pose_params'][:-1])) * 1e4*0.5/interval
                t_rotation=gl_R_6d
                temp_loss += torch.mean(nnfunc.l1_loss(t_rotation[1:] - t_rotation[:-1],t_rotation[1:]*0)) * 1e3*0.5/interval
                temp_loss += torch.mean(nnfunc.l1_loss(T[1:] - T[:-1],T[1:]*0)) * 1e3/interval
                temp_loss += torch.mean(nnfunc.l1_loss(T[1:,2] - T[:-1,2],T[1:,2]*0)) * 5e3/interval
            else:
                temp_loss = 0
            
            reg_loss  = torch.mean(torch.square(batch_flame['expression_params']))*0.5
            reg_loss  = torch.mean(torch.square(batch_flame['shape_params'])) * 5
            reg_loss  += torch.mean(torch.square(batch_flame['jaw_params'][..., 1:])) * 1e4*0.5
            total_loss = landmark_loss2 + reg_loss + temp_loss

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()
            if i_step % 50 == 0:
                t_bar.set_description(f'Batch {batch_id} | Iter {i_step} => Loss {total_loss:.4f} | Landmark Loss {landmark_loss2:.4f} | Reg Loss {reg_loss:.4f} | Temp Loss {temp_loss:.4f}')
            if batch_imgs is not None and i_step%(steps-1)==0: 
                n_imgs = len(batch_imgs)
                img_indices = np.linspace(0, n_imgs - 1, 5, dtype=int)
                save_path = os.path.join(self.saving_root,"visual_results","vis_fit_flame")
                os.makedirs(save_path,exist_ok=True)
                for im_idx in img_indices:
                    _img=batch_imgs[im_idx].clone().numpy().transpose(1,2,0)
                    _t_lmk_mp=t_lmk[im_idx].detach().cpu().numpy()
                    _landmark_mp=landmark_lst_dct[k][im_idx,self.face2d_lmk_distance.selected_mp_indices].detach().cpu().numpy()
                    t_cameras = GS_Camera(R=R[im_idx][None],T=T[im_idx][None],**self.build_cameras_kwargs(1)).to(self.device)
                    mesh_img=self.renderer.render_mesh(ret_dict['vertices'][im_idx,None],cameras=t_cameras)
                    
                    mesh_img  = (mesh_img[:,:3].detach().cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                    mesh_img=cv2.addWeighted(_img,1.0,mesh_img,0.5,0)
                    _img=draw_landmarks(_t_lmk_mp,_img,color=(255, 0, 0))
                    _img=draw_landmarks(_landmark_mp,_img,color=(0, 255, 0))
                    _img = np.concatenate((_img,mesh_img), axis=1)
                    cv2.imwrite(os.path.join(save_path,f"vis_fit_flame_bid-{batch_id}_stp-{i_step}_{im_idx}.png"),  cv2.cvtColor(_img.copy(), cv2.COLOR_RGB2BGR))
                    del _img
                    
        batch_flame['camera_RT_params'].requires_grad=False
        batch_flame['camera_RT_params'][:, :3, 3]=gl_T.detach()
        batch_flame['camera_RT_params'][:, :3, :3]=rotation_6d_to_matrix(gl_R_6d).detach()
        batch_flame['shape_params']=g_flame_shape.expand(batch_size, -1)
        for k, v in batch_flame.items():
            v = v.clone().detach()
        
        eye_pose_code = torch.zeros_like(torch.cat([batch_flame['jaw_params']]*2, dim=1))
        eye_pose_code = torch.nn.Parameter(eye_pose_code, requires_grad=True)

        opt_p = torch.optim.Adam([
            {'params': [eye_pose_code], 'lr': 0.01},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(opt_p, step_size=501, gamma=0.1)

        log('Start tuning FLAME eyepose params')
        t_bar = tqdm(range(steps//2+1), desc='Start tuning FLAME eyepose params')
        for i_step in t_bar:
            batch_flame['eye_pose_params'] = eye_pose_code
            ret_dict = self.flame(batch_flame)
            pred_vertices = ret_dict['vertices']

            pred_lmk_203_gaze, index_gaze_203 = self.flame.reselect_eyes(pred_vertices, 'lmks203')
            pred_lmk_mp_gaze , index_gaze_mp  = self.flame.reselect_eyes(pred_vertices, 'lmks_mp')
            pred_lmk_fan_gaze, index_gaze_fan = self.flame.reselect_eyes(pred_vertices, 'lmks68')

            t_gaze_landmarks = {'lmk_203': pred_lmk_203_gaze, 'lmk_mp': pred_lmk_mp_gaze, 'lmk_fan': pred_lmk_fan_gaze}
            t_gaze_idx = {'lmk_203': index_gaze_203, 'lmk_mp': index_gaze_mp, 'lmk_fan': index_gaze_fan}

            lmk_losses = []
            R = batch_flame['camera_RT_params'][:, :3, :3]
            T = batch_flame['camera_RT_params'][:, :3, 3]
            for k in landmark_lst_dct.keys():
                t_lmk = cameras.transform_points_screen(t_gaze_landmarks[k], R=R, T=T)
                lmk_losses.append(self.face2d_lmk_distance(t_lmk[..., :2], 
                                                           landmark_lst_dct[k][:, t_gaze_idx[k], :2], 
                                                           cam=batch_flame['cam']))
            landmark_loss2 = sum(lmk_losses)
            temp_loss = torch.mean(torch.square(eye_pose_code[1:] - eye_pose_code[:-1])) * 1e6
            total_loss = landmark_loss2 + temp_loss

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()
            scheduler.step()
            if i_step % 50 == 0:
                t_bar.set_description(f'Batch {batch_id} | Iter {i_step} => Loss {total_loss:.4f} | Landmark Loss {landmark_loss2:.4f} | Temp Loss {temp_loss:.4f}')
            if batch_imgs is not None and i_step%(steps//2)==0: 
                n_imgs = len(batch_imgs)
                img_indices = np.linspace(0, n_imgs - 1, 5, dtype=int)
                save_path = os.path.join(self.saving_root,"visual_results","vis_fit_gaze")
                os.makedirs(save_path,exist_ok=True)
                for im_idx in img_indices: 
                    _img=batch_imgs[im_idx].clone().numpy().transpose(1,2,0)
                    _t_lmk_gz=t_lmk[im_idx, :,:2]
                    _landmark_gz=landmark_lst_dct[k][im_idx, t_gaze_idx[k], :2].detach().cpu().numpy()
                    _img=draw_landmarks(_t_lmk_gz,_img,color=(255, 0, 0))
                    _img=draw_landmarks(_landmark_gz,_img,color=(0, 255, 0))

                    cv2.imwrite(os.path.join(save_path,f"vis_fit_gaze_bid-{batch_id}_stp-{i_step}_{im_idx}.png"),  cv2.cvtColor(_img.copy(), cv2.COLOR_RGB2BGR))
                    del _img

        optim_results = {}
        
        for idx, name in enumerate(track_frames):
            optim_results[name] = {
                'expression_params': batch_flame['expression_params'][idx].detach().float().cpu().numpy(),
                'jaw_params': batch_flame['jaw_params'][idx].detach().float().cpu().numpy(),
                'pose_params': batch_flame['pose_params'][idx].detach().float().cpu().numpy(),
                'neck_pose_params': batch_flame['pose_params'][idx].detach().float().cpu().numpy()*0,
                'eye_pose_params': eye_pose_code[idx].detach().float().cpu().numpy(),
                'eyelid_params': batch_flame['eyelid_params'][idx].detach().float().cpu().numpy(),
                'camera_RT_params': batch_flame['camera_RT_params'][idx].detach().float().cpu().numpy(),
                'cam': batch_flame['cam'][idx].detach().float().cpu().numpy(),
            }
        id_share_params_result['flame_shape']=g_flame_shape.detach().float().cpu().numpy()

        return optim_results,id_share_params_result
    
    def run(self, tracked_rlt,id_share_params_result,lmdb_engine=None,frame_interval=1,steps=1001):

        mini_batchs = build_minibatch(list(tracked_rlt.keys()), share_id=True)
        optim_results = {}
        for batch_id, mini_batch in enumerate(mini_batchs):
            mini_batch_face_imgs=None
            if lmdb_engine is not None: mini_batch_face_imgs=[lmdb_engine[f'{key}/head_image'] for key in mini_batch]
            mini_batch_flame_lmk = [tracked_rlt[key] for key in mini_batch]
            mini_batch_flame_lmk = torch.utils.data.default_collate(mini_batch_flame_lmk)
            mini_batch_flame_lmk = data_to_device(mini_batch_flame_lmk, device=self.device)
            optim_result,id_share_params_result = self.optimize(
                mini_batch, mini_batch_flame_lmk,id_share_params_result, share_id=True, batch_id=batch_id, steps=steps,
                batch_imgs=mini_batch_face_imgs,interval=frame_interval
            )
            optim_results.update(optim_result)

        return optim_results,id_share_params_result


def data_to_device(data_dict, device='cuda'):
    assert isinstance(data_dict, dict), 'Data must be a dictionary.'
    for key in data_dict:
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].to(device)
        elif isinstance(data_dict[key], np.ndarray):
            data_dict[key] = torch.tensor(data_dict[key], device=device)
        elif isinstance(data_dict[key], dict):
            data_dict[key] = data_to_device(data_dict[key], device=device)
        else:
            continue
    return data_dict
