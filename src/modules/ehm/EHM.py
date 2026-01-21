import torch
import numpy as np
import torch.nn as nn
from ..mano  import MANO
from ..smplx import SMPLX
from ..flame import FLAME, vertices2landmarks
from ..flame.lbs import lbs, find_dynamic_lmk_idx_and_bcoords,blend_shapes,vertices2joints

class EHM(nn.Module):
    def __init__(self, flame_assets_dir, smplx_assets_dir,mano_assets_dir,
                       n_shape=300, n_exp=50, with_texture=False, 
                       check_pose=True,use_pca=True, num_pca_comps=6, flat_hand_mean=False):
        super().__init__()
        self.smplx = SMPLX(smplx_assets_dir, n_shape=n_shape, n_exp=n_exp, check_pose=check_pose, with_texture=with_texture)
        self.flame = FLAME(flame_assets_dir, n_shape=n_shape, n_exp=n_exp, with_texture=with_texture)
        self.mano  = MANO(mano_assets_dir, use_pca=use_pca, num_pca_comps=num_pca_comps, flat_hand_mean=flat_hand_mean)

        v_template,v_head_template=self.smplx.v_template.clone(),self.flame.v_template.clone()
        tbody_joints = vertices2joints(self.smplx.J_regressor, v_template[None])
        flame_joints = vertices2joints(self.flame.J_regressor, v_head_template[None])
        v_template[self.smplx.smplx2flame_ind]=v_head_template - flame_joints[0, 3:5].mean(dim=0, keepdim=True) + tbody_joints[0, 23:25].mean(dim=0, keepdim=True)
        self.register_buffer('v_template', v_template)

        left_hand_center=self.v_template[self.smplx.smplx2mano_ind['left_hand'],:].mean(0)
        right_hand_center=self.v_template[self.smplx.smplx2mano_ind['right_hand'],:].mean(0)
        self.register_buffer('left_hand_center',left_hand_center)
        self.register_buffer('right_hand_center',right_hand_center)

    def dummy_proj(self, X, camera):
        return X
    
    def forward(self, body_param_dict:dict, flame_param_dict:dict=None,mano_param_dict:dict=None, zero_expression=False, zero_jaw=False, zero_shape=False,
                      proj_type='persp', pose_type='rotmat',):
        
        # for flame head model
        if flame_param_dict is not None:
            eye_pose_params    = flame_param_dict['eye_pose_params']
            shape_params       = flame_param_dict['shape_params']#.clone()
            expression_params  = flame_param_dict['expression_params']#.clone()
            global_pose_params = flame_param_dict.get('pose_params', None)
            jaw_params         = flame_param_dict.get('jaw_params', None)
            eyelid_params      = flame_param_dict.get('eyelid_params', None)
            head_scale         = body_param_dict.get('head_scale', None)
            

            batch_size = shape_params.shape[0]

            # Adjust shape params size if needed
            if shape_params.shape[1] < self.flame.n_shape:
                shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.flame.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
            
            if zero_expression: expression_params = torch.zeros_like(expression_params).to(shape_params.device)
            if zero_jaw: jaw_params = torch.zeros_like(jaw_params).to(shape_params.device)
            if zero_shape: shape_params = torch.zeros_like(shape_params).to(shape_params.device)

            # eye_pose_params  = self.flame.eye_pose.expand(batch_size, -1)
            neck_pose_params = self.flame.neck_pose.expand(batch_size, -1)

            global_pose_params = torch.zeros_like(global_pose_params).to(shape_params.device)
            neck_pose_params = torch.zeros_like(neck_pose_params).to(shape_params.device)

            betas = torch.cat([shape_params, expression_params], dim=1)
            full_pose = torch.cat([global_pose_params, neck_pose_params, jaw_params, eye_pose_params], dim=1)

            template_vertices = self.flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)
            
            head_vertices, head_joints = lbs(betas, full_pose, template_vertices,
                                             self.flame.shapedirs, self.flame.posedirs,
                                             self.flame.J_regressor, self.flame.parents,
                                             self.flame.lbs_weights, dtype=self.flame.dtype)
            
            if eyelid_params is not None:
                head_vertices = head_vertices + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None] #[:, :self.flame.n_ori_verts]
                head_vertices = head_vertices + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]#[:, :self.flame.n_ori_verts]
            if head_scale is not None:
                head_vertices = head_vertices * head_scale[:, None]
        else:
            head_vertices = None

        # body paramerters
        shape_params      = body_param_dict.get('shape')                   # torch.Size([1, 250])
        expression_params = body_param_dict.get('exp', None)               # torch.Size([1, 50])
        global_pose       = body_param_dict.get('global_pose', None)       # torch.Size([1, 1, 3, 3])
        body_pose         = body_param_dict.get('body_pose', None)         # torch.Size([1, 21, 3, 3])
        jaw_pose          = body_param_dict.get('jaw_pose', None)          # torch.Size([1, 1, 3, 3])
        left_hand_pose    = body_param_dict.get('left_hand_pose', None)    # torch.Size([1, 15, 3, 3])
        right_hand_pose   = body_param_dict.get('right_hand_pose', None)   # torch.Size([1, 15, 3, 3])
        eye_pose          = body_param_dict.get('eye_pose', None)          # torch.Size([1, 2, 3, 3])
        joints_offset     = body_param_dict.get('joints_offset',None)
        hand_scale        = body_param_dict.get('hand_scale', None)
        batch_size = shape_params.shape[0]
        
        # if proj_type == 'orth':
        #     projection_func = self.smplx.batch_orth_proj
        # elif proj_type == 'dummy':
        #     projection_func = self.dummy_proj
        # else:
        #     projection_func = self.smplx.batch_week_cam_to_perspective_proj

        if expression_params is None: expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None: global_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if jaw_pose is None: jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        if body_pose is None: body_pose = torch.zeros((batch_size, 21, 3)).to(shape_params.device)
        if len(global_pose.shape) == 2: global_pose = global_pose.unsqueeze(1)
        if len(jaw_pose.shape) == 2: jaw_pose = jaw_pose.unsqueeze(1)

        jaw_pose = torch.zeros((batch_size, 1, 3)).to(shape_params.device)
        eye_pose = torch.zeros((batch_size, 2, 3)).to(shape_params.device)

        if shape_params.shape[-1] < self.smplx.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.smplx.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.smplx.n_shape]
        
        shape_components = torch.cat([t_shape_params, expression_params], dim=1)
        full_pose = torch.cat([global_pose, 
                               body_pose,
                               jaw_pose, 
                               eye_pose,
                               left_hand_pose, 
                               right_hand_pose], dim=1)
        
        template_vertices = self.smplx.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        new_template_vertices = template_vertices + blend_shapes(shape_components, self.smplx.shapedirs)
        tbody_joints = vertices2joints(self.smplx.J_regressor, new_template_vertices)
        if joints_offset is not None: tbody_joints=tbody_joints+joints_offset

        # new_template_vertices, tbody_joints = lbs(shape_components, torch.zeros_like(full_pose), template_vertices,
        #                                           self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
        #                                           self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
        #                                           self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）
        
        if not hasattr(self, 'head_index'): self.head_index = np.unique(self.flame.head_index)
        if head_vertices is not None:
            selected_head = new_template_vertices[:, self.smplx.smplx2flame_ind]
            selected_head[:, self.head_index] = head_vertices[:, self.head_index] - head_joints[:, 3:5].mean(dim=1, keepdim=True) + tbody_joints[:, 23:25].mean(dim=1, keepdim=True)
            new_template_vertices[:, self.smplx.smplx2flame_ind] = selected_head
        
        if hand_scale is not None:
            left_hand_vert = new_template_vertices[:, self.smplx.smplx2mano_ind['left_hand']].clone()
            right_hand_vert = new_template_vertices[:, self.smplx.smplx2mano_ind['right_hand']].clone()
            left_hand_vert = left_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.smplx.left_hand_center[None,None]
            right_hand_vert = right_hand_vert * hand_scale[:, None] + (1-hand_scale[:, None])*self.smplx.right_hand_center[None,None]
            new_template_vertices[:, self.smplx.smplx2mano_ind['left_hand']] = left_hand_vert
            new_template_vertices[:, self.smplx.smplx2mano_ind['right_hand']] = right_hand_vert
            
        vertices, joints = lbs(torch.zeros_like(shape_components), full_pose, new_template_vertices,#
                                            self.smplx.shapedirs, self.smplx.posedirs,        # shapedirs[10475, 3, 20]
                                            self.smplx.J_regressor, self.smplx.parents,       # J_regressor([55, 10475])
                                            self.smplx.lbs_weights,joints_offset=joints_offset, dtype=self.smplx.dtype)   # template_vertices（10475x3）

        head_vert = vertices[:, self.smplx.smplx2flame_ind]
        ret_dict = {}
        landmarksmp = vertices2landmarks(head_vert, self.smplx.flame_faces_tensor,
                                    self.smplx.mp_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                    self.smplx.mp_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        ret_dict['face_lmk_mp'] = landmarksmp
        if self.smplx.using_lmk203:#head_vertices head_vert
            # a=new_template_vertices[:, self.smplx.smplx2flame_ind]
            landmarks203 = vertices2landmarks(head_vert, self.smplx.flame_faces_tensor,
                                        self.smplx.lmk_203_faces_idx.repeat(vertices.shape[0], 1),
                                        self.smplx.lmk_203_bary_coords.repeat(vertices.shape[0], 1, 1))
            ret_dict['face_lmk_203'] = landmarks203
        
        # face dynamic landmarks，lmk_faces_idx（51），lmk_bary_coords（51x3）
        lmk_faces_idx = self.smplx.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.smplx.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
                find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.smplx.dynamic_lmk_faces_idx,
                    self.smplx.dynamic_lmk_bary_coords,
                    self.smplx.head_kin_chain)
            )#dyn_lmk_faces_idx([1, 17])
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.smplx.faces_tensor,
                                       lmk_faces_idx,#faces_tensor([20908, 3])
                                       lmk_bary_coords)
        
        final_joint_set = [joints, landmarks]
        if hasattr(self.smplx, 'extra_joint_selector'):
            # Add any extra joints that might be needed，extra_joints([1, 22, 3])
            extra_joints = self.smplx.extra_joint_selector(vertices, self.smplx.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)
        if self.smplx.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.smplx.extra_joint_regressor, vertices)

            joints[:, self.smplx.source_idxs.long()] = (
                joints[:, self.smplx.source_idxs.long()].detach() * 0.0 +
                reg_joints[:, self.smplx.target_idxs.long()] * 1.0
            )

        landmarks = torch.cat([landmarks[:, -17:], landmarks[:, :-17]], dim=1)

        # save predcition
        prediction = {
            'vertices': vertices,
            'face_kpt': landmarks,
            'joints': joints,
            
            'head_vertices': vertices[:, self.smplx.smplx2flame_ind][:, self.head_index],
            'head_ref_joint': joints[:, 23:25].mean(dim=1, keepdim=True),

            'left_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['left_hand']],
            'left_hand_ref_joint': joints[:, 20:21, :],

            'right_hand_vertices': vertices[:, self.smplx.smplx2mano_ind['right_hand']],
            'right_hand_ref_joint': joints[:, 21:22, :],
        }

        ret_dict.update(prediction)
        
        return ret_dict

    def transform_points3d(self, points3d, M):
        R3d = torch.zeros_like(M)
        R3d[:, :2, :2] = M[:, :2, :2]
        scale = (M[:, 0, 0]**2 + M[:, 0, 1]**2)**0.5
        R3d[:, 2, 2] = scale

        trans = torch.zeros_like(M)[:, 0]
        trans[:, :2] = M[:, :2, 2]
        trans = trans.unsqueeze(1)
        return torch.bmm(points3d, R3d.mT) + trans   # Ugly scale the trans


