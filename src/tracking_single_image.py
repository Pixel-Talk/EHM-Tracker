import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import cv2
import os
import cv2
import torch
import imageio
import numpy as np
import os.path as osp
import math,shutil,json,torchvision
from pytorch3d.renderer import PointLights
from tqdm.auto import tqdm
from PIL import Image
from .utils.graphics import GS_Camera
from .utils.lmdb import LMDBEngine
from .utils.rprint import rlog as log
from .utils.video import images2video
from .modules.dwpose import inference_detector
from .utils.landmark_runner import LandmarkRunner
from .modules.smplx.utils import orginaze_body_pose
from .configs.argument_config import ArgumentConfig
from .modules.renderer.util import weak_cam2persp_cam,cam2persp_cam_fov,cam2persp_cam_fov_body
from .modules.refiner.flame_refiner import FlameOptimizer
from .modules.refiner.ehm_refiner import EhmOptimizer
from .configs.data_prepare_config import DataPreparationConfig
from .utils.io import load_config, write_dict_pkl, load_dict_pkl
from .utils.crop import crop_image, parse_bbox_from_landmark_lite, crop_image_by_bbox, _transform_pts
from .utils.helper import load_onnx_model, instantiate_from_config, image2tensor, image2tensor, get_machine_info
from .modules.refiner.smplx_utils import smplx_joints_to_dwpose

class DataPreparePipeline(object):
    def __init__(self, data_prepare_cfg: DataPreparationConfig):
        self.cfg = data_prepare_cfg
        self.device = self.cfg.device
        self.dwpose_detector = instantiate_from_config(load_config(self.cfg.dwpose_cfg_path))
        self.dwpose_detector.warmup()
        self.pixie_encoder = instantiate_from_config(load_config(self.cfg.pixie_cfg_path))
        self.pixie_encoder.to(self.device)
        self.matte = instantiate_from_config(load_config(self.cfg.matting_cfg_path))
        self.matte.to(self.device)

        self.lmk70_detector = load_onnx_model(load_config(self.cfg.kp70_cfg_path)) # !!will force lmk in 256x256 resolution
        self.mp_detector   = instantiate_from_config(load_config(self.cfg.mp_cfg_path))
        self.mp_detector.warmup()
        self.teaser_encoder = load_onnx_model(load_config(self.cfg.teaser_cfg_path))
        self.hamer_encoder = load_onnx_model(load_config(self.cfg.hamer_cfg_path))
        self.landmark_runner = LandmarkRunner(ckpt_path=self.cfg.kp203_path, onnx_provider=self.device)
        self.landmark_runner.warmup()

        self.flame_opt = FlameOptimizer(self.cfg.flame_assets_dir, device=self.device, image_size=self.cfg.head_crop_size,tanfov=self.cfg.tanfov)
        self.ehm_opt   = EhmOptimizer(self.cfg.flame_assets_dir, self.cfg.smplx_assets_dir, self.cfg.mano_assets_dir, 
                                        device=self.device, body_image_size=self.cfg.body_hd_size, head_image_size=self.cfg.head_crop_size,tanfov=self.cfg.tanfov,
                                        vposer_ckpt=self.cfg.vposer_ckpt_dir)
        
        self.flame = self.flame_opt.flame
        self.ehm_opt.ehm.flame = self.flame                      # for saving memory purpose
        self.ehm_opt.head_renderer = self.flame_opt.renderer     # for saving memory purpose
        self.ehm = self.ehm_opt.ehm
        self.head_renderer = self.flame_opt.renderer
        self.body_renderer = self.ehm_opt.body_renderer

    def get_image_name(self, image_fp, using_last_k=1):
        aa_names  = []
        sub_names = image_fp.split(os.sep)
        for ii, xx in enumerate(sub_names):
            if ii >= len(sub_names) - using_last_k:
                aa_names.append(xx)
        if aa_names[0].startswith('__'): aa_names[0] = aa_names[0][2:]
        if aa_names[-1].endswith('.png') or aa_names[-1].endswith('.jpg'):
            aa_names[-1] = aa_names[-1].split('.')[0]
        return '__'.join(aa_names)

    def get_union_human_box_in_video(self, video_fp,frame_interval=1):
        reader = imageio.get_reader(video_fp)
        num_frames = reader.count_frames()
        ## get union bbox' w and h
        union_bbox = []
        for idx in tqdm(range(0,num_frames,frame_interval),desc='Getting unioned bbox ...'):
            img_rgb=reader.get_data(idx)
            try:
                bbox = inference_detector(self.dwpose_detector.pose_estimation.session_det, img_rgb)[0]
                union_bbox.append(bbox)
            except Exception as e:
                pass
        
        if len(union_bbox) == 0:
            return None

        uu = np.array(union_bbox)
        ul, ut, ur, ub = uu[:, 0].min(), uu[:, 1].min(), uu[:, 2].max(), uu[:, 3].max()
        ucx, ucy = (ul + ur) / 2, (ut + ub) / 2
        usize = max(ur - ul, ub - ut)
        union_box = [ucx - usize / 2, ucy - usize / 2, ucx + usize / 2, ucy + usize / 2]

        return union_box

    def cvt_weak_cam_to_persp_cam(self, wcam):
        R, T = weak_cam2persp_cam(wcam, focal_length=self.cfg.focal_length, z_dist=self.cfg.z_dist)
        return torch.cat((R, T[..., None]), axis=-1)
    def cvt_cam_to_persp_cam_fov(self, wcam):
        R, T = cam2persp_cam_fov(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    def cvt_cam_to_persp_cam_fov_body(self, wcam):
        R, T = cam2persp_cam_fov_body(wcam, tanfov=self.cfg.tanfov)
        return torch.cat((R, T[..., None]), axis=-1)
    
    def track_base(self, img_rgb, union_box=None, last_results=None):
        ret_images = {}
        base_results = {}
        mean_shape_results = {}

        ret_images[f'ori_image'] = img_rgb

        det_info, det_raw_info = self.dwpose_detector(img_rgb)
        if union_box is not None: det_info['bbox'] = union_box
        if det_info['bbox'] is None: 
            print("          Missing box")
            return None, None, None
        crop_info_hd = crop_image_by_bbox(img_rgb, det_info['bbox'], dsize=self.cfg.body_hd_size) 
        crop_info  = crop_image_by_bbox(img_rgb, det_info['bbox'],   dsize=self.cfg.body_crop_size)
        base_results['body_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o'], 
                                     'M_o2c-hd': crop_info_hd['M_o2c'], 'M_c2o-hd': crop_info_hd['M_c2o']}
        base_results['dwpose_raw'] = det_raw_info
        base_results['dwpose_rlt'] = {'keypoints': _transform_pts(det_raw_info['keypoints'], crop_info_hd['M_o2c']), 
                                      'scores': det_raw_info['scores'], 'faces': det_info['faces'], 'hands': det_info['hands']}
        
        img_crop = crop_info['img_crop']
        img_hd   = crop_info_hd['img_crop']
        ret_images['body_image'] = img_hd

        img_crop = image2tensor(img_crop).to(self.device).unsqueeze(0)
        img_hd = image2tensor(img_hd).to(self.device).unsqueeze(0)

        # matting related
        t_matting = self.matte(img_hd.contiguous(), 'alpha')
        ret_images['body_mask'] = (np.clip(t_matting.cpu().numpy(), 0, 1) * 255).round().astype(np.uint8)
        predict = t_matting.expand(3, -1, -1)
        matting_image = img_hd.clone()[0]
        background_rgb = 1.0
        background_rgb = matting_image.new_ones(matting_image.shape) * background_rgb
        matting_image = matting_image * predict + (1-predict) * background_rgb
        ret_images['body_matting'] =  np.transpose(np.clip(matting_image.cpu().numpy(), 0, 1) * 255,(1,2,0)).round().astype(np.uint8)
        img_hd = matting_image.unsqueeze(0)
        img_crop = torch.nn.functional.interpolate(img_hd, (224, 224))
        # body related
        coeff_param = self.pixie_encoder(img_crop, img_hd)['body']
        coeff_param = orginaze_body_pose(coeff_param)
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov_body(coeff_param['body_cam'])
        coeff_param = {k: v.cpu().numpy() for k, v in coeff_param.items()}
        base_results['smplx_coeffs'] = coeff_param
        mean_shape_results['smplx_shape'] = coeff_param['shape']

        # head related
        crop_info  = crop_image(img_rgb, det_info['faces'], dsize=self.cfg.head_crop_size, scale=1.75) 
        base_results['head_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        ret_images['head_image'] = crop_info['img_crop']
        
        lmk203 = self.landmark_runner.run(crop_info['img_crop'])['pts']
        t_img = crop_info['img_crop'].transpose((2, 0, 1)).astype(np.float32)
        lmk70 = self.lmk70_detector.run(t_img[None]/255.)['pts'] * 2
        lmk_mp = self.mp_detector.run(crop_info['img_crop'])['pts']
        
        if lmk203 is None or lmk_mp is None or lmk70 is None:
            if last_results is not None:
                base_results.update({'head_lmk_203': last_results['head_lmk_203'], 
                                     'head_lmk_70':  last_results['head_lmk_70'], 
                                     'head_lmk_mp':  last_results['head_lmk_mp']})
            else:
                return None, None, None
        else:
            if len(lmk203.shape) == 3: lmk203 = lmk203[0]
            if len(lmk70.shape) == 3: lmk70 = lmk70[0]
            if len(lmk_mp.shape) == 3: lmk_mp = lmk_mp[0]

            base_results.update({'head_lmk_203': lmk203, 'head_lmk_70': lmk70, 'head_lmk_mp': lmk_mp})

        cropped_image = cv2.resize(crop_info['img_crop'], (self.cfg.teaser_input_size, self.cfg.teaser_input_size))
        cropped_image = np.transpose(cropped_image, (2,0,1))[None, ...] / 255.0
        coeff_param = self.teaser_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['cam'])).numpy()
        base_results.update({'flame_coeffs': coeff_param})
        mean_shape_results['flame_shape'] = coeff_param['shape_params']

        # hand related
        all_hands_kps = det_info['hands']
        hand_kps_l = all_hands_kps[0]
        hand_kps_r = all_hands_kps[1]
        crop_info  = crop_image_by_bbox(img_rgb, parse_bbox_from_landmark_lite(hand_kps_l)['bbox'], 
                                        lmk=hand_kps_l, scale=1.3,
                                        dsize=self.cfg.hand_crop_size) 
        ret_images['left_hand_image'] = crop_info['img_crop']
        is_left = True
        t_img = cv2.flip(crop_info['img_crop'], 1) if is_left else crop_info['img_crop']
        cropped_image = np.transpose(t_img, (2,0,1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['pred_cam'])).numpy()
        base_results.update({'left_mano_coeffs': coeff_param})
        base_results.update({'left_hand_crop': {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}})
        mean_shape_results['left_mano_shape'] = coeff_param['betas']

        crop_info  = crop_image_by_bbox(img_rgb, parse_bbox_from_landmark_lite(hand_kps_r)['bbox'], 
                                        lmk=hand_kps_r, scale=1.3,
                                        dsize=self.cfg.hand_crop_size) 
        ret_images['right_hand_image'] = crop_info['img_crop']
        cropped_image = np.transpose(crop_info['img_crop'], (2,0,1))[None, ...] / 255.0
        coeff_param = self.hamer_encoder(cropped_image.astype(np.float32))
        coeff_param['camera_RT_params'] = self.cvt_cam_to_persp_cam_fov(torch.from_numpy(coeff_param['pred_cam'])).numpy()
        base_results.update({'right_mano_coeffs': coeff_param})
        base_results.update({'right_hand_crop': {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}})
        mean_shape_results['right_mano_shape'] = coeff_param['betas']

        return ret_images, base_results, mean_shape_results

    def execute(self, args: ArgumentConfig):
        out_dir = args.output_dir
        log(str(get_machine_info()))
        self.args=args
        ######## process driving info ########
        for image_idx, image_fp in enumerate(args.input_dir):
            image_name = self.get_image_name(image_fp, 1)
            saving_root = os.path.join(out_dir, image_name)
            out_image_fp=image_fp
            out_lmdb_dir = os.path.join(saving_root, 'img_lmdb')
            id_share_params_fp = osp.join(saving_root, 'id_share_params.pkl')
            base_track_fp = os.path.join(saving_root, 'base_tracking.pkl')
            skipped_flag = os.path.join(saving_root, f"skipped.txt")
            optim_track_fp_flame = os.path.join(saving_root, 'optim_tracking_flame.pkl')
            optim_track_fp_smplx = os.path.join(saving_root, 'optim_tracking_ehm.pkl')
            image_info_path=os.path.join(saving_root, 'videos_info.json')
            os.makedirs(saving_root,exist_ok=True)
            if os.path.exists(skipped_flag) :
                log("Exist skipping flag, Skipping!")
                continue
            if os.path.exists(optim_track_fp_smplx):
                log("Tracking results already esist !")
                if not args.save_vis_video: continue
            
            try:
                __image=Image.open(image_fp)
                os.makedirs(os.path.join(saving_root,'images'),exist_ok=True)
                __image.save(os.path.join(saving_root,'images', f'{os.path.basename(image_fp)}'))
                img_rgb=np.array(__image,dtype=np.uint8)
                img_rgb=img_rgb[...,:3]
                num_frames = 1
                
                if not (self.cfg.check_skip_extraction and osp.exists(base_track_fp)):

                    log(f"[{image_idx:04d}/{len(args.input_dir)}] Processing imge file: {out_image_fp}")

                    union_box = None
                    base_results = {}
                    id_share_params_results = {} # like shape params 
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)

                    with torch.no_grad():
                        last_results = None     # for bad frame tracking
                        img_idx = 0
                        for idx in tqdm(range(0,num_frames),desc='Extracting feature ...',total=num_frames):
                            
                            b_name = f'frame_{img_idx:06d}'
                            ret_images, ret_results, shape_results = self.track_base(img_rgb, union_box, last_results)
                            last_results = ret_results

                            if ret_results is None:
                                log(f"Skipping {image_fp} due to incomplete facial landmark extraction")
                                continue

                            for k, v in ret_images.items():
                                if len(v.shape) == 2: 

                                    v = v[:,:,None]#cv2.merge([v]*3)
                                lmdb_engine.dump(f'{b_name}/{k}', payload=image2tensor(v, norm=False), type='image')
                            
                            for k, v in ret_images.items():
                                if len(v.shape) == 2: 
                                    v = v[:,:,None]
                                lmdb_engine.dump(f'{b_name}/{k}', payload=image2tensor(v, norm=False), type='image')
                            
                            del ret_results['flame_coeffs']['shape_params'] #shape
                            del ret_results['smplx_coeffs']['shape'] #shape
                            base_results[b_name] = ret_results
                            
                        # merge mean shape results
                            for k, v in shape_results.items():
                                if k not in id_share_params_results: id_share_params_results[k] = []
                                id_share_params_results[k].append(v)
                            img_idx+=1
                            
                    for k, v in id_share_params_results.items():
                        id_share_params_results[k] = np.array(v).mean(0)
                        

                    write_dict_pkl(id_share_params_fp, id_share_params_results)
                    write_dict_pkl(base_track_fp, base_results)
                    lmdb_engine.random_visualize(os.path.join(out_lmdb_dir, 'visualize.jpg'))
                    lmdb_engine.close()
                else:
                    base_results = load_dict_pkl(base_track_fp)
                    id_share_params_results= load_dict_pkl(id_share_params_fp)
                    
                # optimize head parameters
                optimized_result = base_results     # init
                if os.path.exists(optim_track_fp_flame):
                    optimized_result = load_dict_pkl(optim_track_fp_flame)
                    
                else:
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    if self.cfg.fit_flame:
                        self.flame_opt.saving_root=saving_root
                        log(f"[{image_idx:04d}/{len(args.input_dir)}] Refining head parameters: {image_name}")
                        opt_flame_coeff,id_share_params_results = self.flame_opt.run(optimized_result,id_share_params_results,lmdb_engine,steps=201)
                        for key in base_results.keys():
                            base_results[key]['flame_coeffs'] = opt_flame_coeff[key]
                        write_dict_pkl(optim_track_fp_flame, base_results)
                        optimized_result=base_results
                    lmdb_engine.close()
                    
                # optimize ehm parameters
                if os.path.exists(optim_track_fp_smplx):
                    optimized_result = load_dict_pkl(optim_track_fp_smplx)
                else:
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    if self.cfg.fit_ehm:
                        self.ehm_opt.saving_root=saving_root
                        log(f"[{image_idx:04d}/{len(args.input_dir)}] Refining ehm-smplx parameters: {image_name}")
                        opt_smplx_coeff,id_share_params_results = self.ehm_opt.run(optimized_result,id_share_params_results,lmdb_engine,steps=101)
                        for key in base_results.keys():
                            optimized_result[key]['smplx_coeffs'] = opt_smplx_coeff[key]
                            del optimized_result[key]['left_mano_coeffs']['betas'] #shape
                            del optimized_result[key]['right_mano_coeffs']['betas'] #shape
                            

                        write_dict_pkl(optim_track_fp_smplx, optimized_result)
                        write_dict_pkl(id_share_params_fp, id_share_params_results)
                    lmdb_engine.close()
                
                frames_key=list(optimized_result.keys())
                image_info={image_name:{"frames_num":len(frames_key),"frames_keys":frames_key}}
                with open(image_info_path, 'w', encoding='utf-8') as json_file:
                    json.dump(image_info, json_file, ensure_ascii=False, indent=4)
                
                if args.save_vis_video:
                    track_video_fp = os.path.join(saving_root, 'viz_tracking.mp4')
                    if not os.path.exists(track_video_fp):
                        lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                        all_images = []
                        device = self.cfg.device
                        cameras_kwargs = self.ehm_opt.build_cameras_kwargs(1,self.ehm_opt.body_focal_length)
                    
                        
                        with torch.no_grad():
                            lights=PointLights(device=self.device, location=[[0.0, -1.0, -100.0]])

                            for idx, image_key in tqdm(enumerate(optimized_result.keys()), desc='Saving visualized results', total=len(optimized_result)):

                                t_flame_coeffs,t_smplx_coeffs,t_left_mano_coeffs,t_right_mano_coeffs=self.convert_traking_params(optimized_result,id_share_params_results,image_key,device)
                                #ret_body = self.ehm_opt.ehm(t_smplx_coeffs, t_flame_coeffs, {'left_hand': t_left_mano_coeffs, 'right_hand': t_right_mano_coeffs})
                                ret_body = self.ehm_opt.ehm(t_smplx_coeffs, t_flame_coeffs)

                                
                                xx=ret_body['vertices']
                                camera_RT_params=torch.tensor(optimized_result[image_key]['smplx_coeffs']['camera_RT_params']).to(device)
                                R, T = camera_RT_params.split([3, 1], dim=-1)
                                T = T.squeeze(-1)
                                R,T=R[None],T[None]
                                #cameras = PerspectiveCameras(R=R,T=T,**cameras_kwargs).to(device)
                                cameras = GS_Camera(R=R,T=T,**cameras_kwargs).to(device)
                                proj_joints   = cameras.transform_points_screen(ret_body['joints'], R=R, T=T)
                                pred_kps2d = smplx_joints_to_dwpose(proj_joints)[0][..., :2]
                                gt_lmk_2d=optimized_result[image_key]['dwpose_rlt']['keypoints']
                                rendered_img = self.body_renderer.render_mesh(xx,cameras,lights=lights,smplx2flame_ind=self.ehm_opt.ehm.smplx.smplx2flame_ind)
                                t_img  = (rendered_img[:,:3].cpu().numpy()).clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                                if args.save_visual_render:
                                    
                                    os.makedirs(os.path.join(saving_root, 'mesh_rendered'), exist_ok=True)
                                    rendered_img=rendered_img.cpu().numpy().clip(0, 255).astype(np.uint8)[0].transpose(1,2,0)
                                    img_ = Image.fromarray(rendered_img)
                                    img_.save(os.path.join(saving_root, 'mesh_rendered',f"{str(idx).zfill(5)}.png"))
                                img_inp = lmdb_engine[f'{image_key}/body_image'].numpy().transpose(1,2,0)
                                img_bld = cv2.addWeighted(img_inp, 0.5, t_img, 0.5, 1)
                                img_ret = cv2.hconcat([img_inp, t_img, img_bld])
                                # Draw predicted keypoints in red
                                for kp in pred_kps2d[0].cpu().numpy():
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x,y), 3, (0,0,255), -1)  # Red color (BGR format)
                                    
                                # Draw ground truth keypoints in green 
                                for kp in gt_lmk_2d:
                                    x, y = int(kp[0]), int(kp[1])
                                    cv2.circle(img_bld, (x,y), 3, (0,255,0), -1)  # Green color (BGR format)
                                all_images.append(img_ret)
                                
                                    
                        lmdb_engine.random_visualize(os.path.join(out_lmdb_dir, 'visualize.jpg'))
                        lmdb_engine.close()
                        images2video(all_images, track_video_fp, fps=30)

                    log(f'Tacking results is saved {image_name[-20:]} ==> {track_video_fp}')
                    
            except Exception as e:
                
                log(f"Skipping {out_image_fp} due to error {e}")
                import traceback
                log(f"Traceback:{traceback.format_exc()}")
                skipped_flag = os.path.join(saving_root, f"skipped.txt")
                if os.path.isdir(skipped_flag):
                    try:
                        shutil.rmtree(skipped_flag)
                        print(f"Folder {skipped_flag} has been deleted.")
                    except Exception as e:
                        print(f"Failed to delete folder {skipped_flag}: {e}")

                
                try: 
                    lmdb_engine.close()
                except:
                    pass
                
                continue
            

    def convert_traking_params(self,optimized_result,id_share_params_results,image_key,device):
        t_flame_coeffs = {k:torch.from_numpy(v)[None].to(device) for k, v in optimized_result[image_key]['flame_coeffs'].items()}
        t_smplx_coeffs = {k:torch.from_numpy(v)[None].to(device) for k, v in optimized_result[image_key]['smplx_coeffs'].items()}
        t_left_mano_coeffs = {k:torch.from_numpy(v).to(device) for k, v in optimized_result[image_key]['left_mano_coeffs'].items()}
        t_right_mano_coeffs = {k:torch.from_numpy(v).to(device) for k, v in optimized_result[image_key]['right_mano_coeffs'].items()}
        t_smplx_coeffs["shape"],t_flame_coeffs["shape_params"]=torch.from_numpy(id_share_params_results["smplx_shape"]).to(device),torch.from_numpy(id_share_params_results["flame_shape"]).to(device)
        t_smplx_coeffs["joints_offset"]=torch.from_numpy(id_share_params_results["joints_offset"]).to(device)
        
        t_smplx_coeffs["head_scale"]=torch.from_numpy(id_share_params_results["head_scale"]).to(device)
        t_smplx_coeffs["hand_scale"]=torch.from_numpy(id_share_params_results["hand_scale"]).to(device)
        
        t_left_mano_coeffs["betas"],t_right_mano_coeffs["betas"]=torch.from_numpy(id_share_params_results["left_mano_shape"]).to(device),torch.from_numpy(id_share_params_results["right_mano_shape"]).to(device)

        return t_flame_coeffs,t_smplx_coeffs,t_left_mano_coeffs,t_right_mano_coeffs



    def del_extra_params_values(optim_frame_params):
        #dict_keys(['body_crop', 'dwpose_raw', 'dwpose_rlt', 'smplx_coeffs', 'head_crop', 'head_lmk_203', 'head_lmk_70', 
        # 'head_lmk_mp', 'flame_coeffs','left_mano_coeffs', 'left_hand_crop', 'right_mano_coeffs', 'right_hand_crop'])
        del optim_frame_params['body_crop']
        del optim_frame_params['dwpose_raw']
        del optim_frame_params['dwpose_rlt']
        del optim_frame_params['head_crop']
        del optim_frame_params['head_lmk_203']
        del optim_frame_params['head_lmk_70']
        del optim_frame_params['head_lmk_mp']
        del optim_frame_params['left_hand_crop']
        del optim_frame_params['right_hand_crop']
        return optim_frame_params        
        
import argparse
class Local_Args:
    input_dir=''
    output_dir=''
    save_vis_video=True
    save_visual_render=True
    
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('-i','--input_dir',type=str,nargs='+',help='source image file path')
    parser.add_argument('-o','--output_dir',type=str,help='output directory')
    parser.add_argument('--save_vis_video',default=False,action='store_true',help='save tracking video')
    parser.add_argument('--save_visual_render',default=False,action='store_true',help='save visual render')
    args=parser.parse_args()
    local_args=Local_Args()
    local_args.output_dir=args.output_dir
    local_args.save_vis_video=args.save_vis_video
    local_args.save_visual_render=args.save_visual_render
    
    data_preprocesser=DataPreparePipeline(DataPreparationConfig())
    for sdir in args.input_dir:
        
        if os.path.isfile(sdir):
            local_args.input_dir=sdir
            data_preprocesser.execute(args)
        elif os.path.isdir(sdir):
            local_args.input_dir=[os.path.join(sdir,xx) for xx in os.listdir(sdir)]
            data_preprocesser.execute(local_args)
        else:
            print(f"Invalid source path:{sdir}")
            continue
    print("Done!")
            
    