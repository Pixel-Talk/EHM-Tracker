import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import os.path as osp
from .lbs import Struct, to_tensor, to_np, lbs, vertices2landmarks, JointsFromVerticesSelector, find_dynamic_lmk_idx_and_bcoords
from ...utils import rotation_converter as converter
from ...utils.rprint import rlog as log

## SMPLX 
J14_NAMES = [
    'right_ankle',
    'right_knee',
    'right_hip',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_wrist',
    'right_elbow',
    'right_shoulder',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'neck',
    'head',
]
SMPLX_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplx', 'right_eye_smplx', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3', 'right_eye_brow1', 'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1', 'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1', 'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2', 'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1', 'left_mouth_5', 'left_mouth_4', 'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3', 'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1', 'head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
extra_names = ['head_top', 'left_big_toe', 'left_ear', 'left_eye', 'left_heel', 'left_index', 'left_middle', 'left_pinky', 'left_ring', 'left_small_toe', 'left_thumb', 'nose', 'right_big_toe', 'right_ear', 'right_eye', 'right_heel', 'right_index', 'right_middle', 'right_pinky', 'right_ring', 'right_small_toe', 'right_thumb']
SMPLX_names += extra_names

part_indices = {}
part_indices['body'] = np.array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
                                13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24, 123,
                                124, 125, 126, 127, 132, 134, 135, 136, 137, 138, 143])
part_indices['torso'] = np.array([  0,   1,   2,   3,   6,   9,  12,  13,  14,  15,  16,  17,  18,
                                19,  22,  23,  24,  55,  56,  57,  58,  59,  76,  77,  78,  79,
                                80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                                93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                            119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                            132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144])
part_indices['head'] = np.array([ 12,  15,  22,  23,  24,  55,  56,  57,  58,  59,  60,  61,  62,
                                63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
                                76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
                                89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
                            102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                            115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 134, 136,
                            137])
part_indices['face'] = np.array([ 55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['upper'] = np.array([ 12, 13, 14, 55, 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
                            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                            80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
                            93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105,
                        106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
                        119, 120, 121, 122])
part_indices['hand'] = np.array([ 20,  21,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
                        36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,
                        49,  50,  51,  52,  53,  54, 128, 129, 130, 131, 133, 139, 140,
                        141, 142, 144])
part_indices['left_hand'] = np.array([ 20,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
                        37,  38,  39, 128, 129, 130, 131, 133])
part_indices['right_hand'] = np.array([ 21,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                        52,  53,  54, 139, 140, 141, 142, 144])
# kinematic tree 
head_kin_chain = [15,12,9,6,3,0]

#--smplx joints
# 00 - Global
# 01 - L_Thigh
# 02 - R_Thigh
# 03 - Spine
# 04 - L_Calf
# 05 - R_Calf
# 06 - Spine1
# 07 - L_Foot
# 08 - R_Foot
# 09 - Spine2
# 10 - L_Toes
# 11 - R_Toes
# 12 - Neck
# 13 - L_Shoulder
# 14 - R_Shoulder
# 15 - Head
# 16 - L_UpperArm
# 17 - R_UpperArm
# 18 - L_ForeArm
# 19 - R_ForeArm
# 20 - L_Hand
# 21 - R_Hand
# 22 - Jaw
# 23 - L_Eye
# 24 - R_Eye

class SMPLX(nn.Module):
    """
    Given smplx parameters, this class generates a differentiable SMPLX function
    which outputs a mesh and 3D joints
    """
    def __init__(self, smplx_assets_dir, n_shape=200, n_exp=50, with_texture=False, check_pose=True):
        super(SMPLX, self).__init__()
        log("creating the SMPLX Decoder")
        self.n_shape = n_shape
        self.check_pose = check_pose
        self.with_texture = with_texture
        smplx_model_path = osp.join(smplx_assets_dir, 'SMPLX_NEUTRAL_2020.npz')
        ss = np.load(smplx_model_path, allow_pickle=True)
        smplx_model = Struct(**ss)

        flame_model_path = osp.join(smplx_assets_dir, 'flame_generic_model.pkl')
        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(smplx_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('flame_faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(smplx_model.v_template), dtype=self.dtype))
        # The shape components and expression
        # expression space is the same as FLAME
        shapedirs = to_tensor(to_np(smplx_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = smplx_model.posedirs.shape[-1]
        posedirs = np.reshape(smplx_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype)) 
        self.register_buffer('J_regressor', to_tensor(to_np(smplx_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(smplx_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(smplx_model.weights), dtype=self.dtype))
        # for face keypoints
        self.register_buffer('lmk_faces_idx', torch.tensor(smplx_model.lmk_faces_idx, dtype=torch.long))
        self.register_buffer('lmk_bary_coords', torch.tensor(smplx_model.lmk_bary_coords, dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', torch.tensor(smplx_model.dynamic_lmk_faces_idx, dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', torch.tensor(smplx_model.dynamic_lmk_bary_coords, dtype=self.dtype))
        # pelvis to head, to calculate head yaw angle, then find the dynamic landmarks
        self.register_buffer('head_kin_chain', torch.tensor(head_kin_chain, dtype=torch.long))

        #-- initialize parameters 
        # shape and expression
        self.register_buffer('shape_params', nn.Parameter(torch.zeros([1, n_shape], dtype=self.dtype), requires_grad=False))
        self.register_buffer('expression_params', nn.Parameter(torch.zeros([1, n_exp], dtype=self.dtype), requires_grad=False))
        # pose: represented as rotation matrx [number of joints, 3, 3]
        self.register_buffer('global_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('head_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('neck_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('jaw_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1,1,1), requires_grad=False))
        self.register_buffer('eye_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(2,1,1), requires_grad=False))
        self.register_buffer('body_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(21,1,1), requires_grad=False))
        self.register_buffer('left_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))
        self.register_buffer('right_hand_pose', nn.Parameter(torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15,1,1), requires_grad=False))

        extra_joint_path = osp.join(smplx_assets_dir, 'smplx_extra_joints.yaml')
        if osp.exists(extra_joint_path):
            self.extra_joint_selector = JointsFromVerticesSelector(
                fname=extra_joint_path)
        self.use_joint_regressor = True
        self.keypoint_names = SMPLX_names
        if self.use_joint_regressor:
            j14_regressor_path = osp.join(smplx_assets_dir, 'SMPLX_to_J14.pkl')
            with open(j14_regressor_path, 'rb') as f:
                j14_regressor = pickle.load(f, encoding='latin1')
            source = []
            target = []
            for idx, name in enumerate(self.keypoint_names):
                if name in J14_NAMES:
                    source.append(idx)
                    target.append(J14_NAMES.index(name))
            source = np.asarray(source)
            target = np.asarray(target)
            self.register_buffer('source_idxs', torch.from_numpy(source))
            self.register_buffer('target_idxs', torch.from_numpy(target))
            joint_regressor = torch.from_numpy(
                j14_regressor).to(dtype=torch.float32)
            self.register_buffer('extra_joint_regressor', joint_regressor)
            self.part_indices = part_indices
        
        self.smplx2flame_ind = np.load(osp.join(smplx_assets_dir, 'SMPL-X__FLAME_vertex_ids.npy'))
        self.register_buffer('face_l_eyelid', torch.from_numpy(np.load(osp.join(smplx_assets_dir, 'flame_l_eyelid.npy'))).to(self.dtype)[None])
        self.register_buffer('face_r_eyelid', torch.from_numpy(np.load(osp.join(smplx_assets_dir, 'flame_r_eyelid.npy'))).to(self.dtype)[None])

        lmk_embeddings_mp = np.load(osp.join(smplx_assets_dir, "mediapipe_landmark_embedding.npz"))
        self.register_buffer('mp_lmk_faces_idx', torch.from_numpy(lmk_embeddings_mp['lmk_face_idx'].astype('int32')).long())
        self.register_buffer('mp_lmk_bary_coords', torch.from_numpy(lmk_embeddings_mp['lmk_b_coords']).to(self.dtype))
        self.lmk_mp_indices = lmk_embeddings_mp['landmark_indices'].tolist()

        with open(osp.join(smplx_assets_dir, 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as fid:
            self.smplx2mano_ind = pickle.load(fid, encoding='latin1')

        self.using_lmk203 = False
        lmk203_path = osp.join(smplx_assets_dir, "203_landmark_embeding.npz")
        if osp.exists(lmk203_path):
            self.using_lmk203 = True
            lmk_embeddings_203 = np.load(lmk203_path)
            self.register_buffer('lmk_203_faces_idx', torch.from_numpy(lmk_embeddings_203['lmk_face_idx'].astype('int32')).long())
            self.register_buffer('lmk_203_bary_coords', torch.from_numpy(lmk_embeddings_203['lmk_b_coords']).to(self.dtype))
            self.lmk_203_front_indices = lmk_embeddings_203['landmark_front_indices'].tolist()
            self.lmk_203_left_indices  = lmk_embeddings_203['landmark_left_indices']
            self.lmk_203_right_indices = lmk_embeddings_203['landmark_right_indices']
        left_hand_center=self.v_template[self.smplx2mano_ind['left_hand'],:].mean(0)
        right_hand_center=self.v_template[self.smplx2mano_ind['right_hand'],:].mean(0)
        self.register_buffer('left_hand_center', left_hand_center)
        self.register_buffer('right_hand_center', right_hand_center)
    def batch_orth_proj(self, X, camera):
        '''
            X is N x num_verts x 3
            camera is N x 3, each stands for s, x, y
        '''
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
        Xn = (camera[:, :, 0:1] * X_trans)
        return Xn

    def batch_week_cam_to_perspective_proj(self, X, camera):
        '''
            X is B x num_verts x 3
            Render camera is at [0, 0, 2] and look at [0, 0, 0]
        '''
        camera = camera.clone().view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:3]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)

        X, Y, Z = X_trans[..., 0:1], X_trans[..., 1:2], X_trans[..., 2:]

        new_Z = Z + 2
        # Z0 = new_Z.mean(1).unsqueeze(-1)
        Z0 = 2

        ss = camera[:, :, 0:1]

        ff = ss * Z0
        scale_perspective = ff / new_Z

        X_perspective = scale_perspective * X
        Y_perspective = scale_perspective * Y
        Z_new = Z * ss

        Xn = torch.cat([X_perspective, Y_perspective, Z_new], dim=-1)
        return Xn

    def batch_persp_proj(self, X, camera):
        '''
            X is B x num_verts x 3
            Render camera is B x 6, each stands for [s, tx, ty, dx, dy, dz], dx, dy, dz are correction param
        '''
        camera = camera.clone().view(-1, 1, 6)
        X_trans = X[:, :, :2] + camera[:, :, 1:3]
        X_trans = torch.cat([X_trans, X[:,:,2:]], 2)

        X, Y, Z = X_trans[..., 0:1], X_trans[..., 1:2], X_trans[..., 2:]

        dx, dy, dz = camera[..., 3:4], camera[..., 4:5], camera[..., 5:6]

        new_Z = Z + 2 + dz
        # Z0 = new_Z.mean(1).unsqueeze(-1)
        Z0 = 2 + dz

        ss = camera[:, :, 0:1]

        ff = ss * Z0
        scale_perspective = ff / new_Z

        X_perspective = scale_perspective * X + dx
        Y_perspective = scale_perspective * Y + dy
        Z_new = Z * ss

        Xn = torch.cat([X_perspective, Y_perspective, Z_new], dim=-1)
        return Xn
    
    def pose_abs2rel(self, global_pose, body_pose, abs_joint = 'head'):
        ''' change absolute pose to relative pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        '''
        if abs_joint == 'head':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == 'neck':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == 'right_wrist':
            # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
            # -> right elbow -> right wrist
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == 'left_wrist':
            # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
            # -> Left elbow -> Left wrist
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(
                f'pose_abs2rel does not support: {abs_joint}')

        batch_size = global_pose.shape[0]
        dtype = global_pose.dtype
        device = global_pose.device
        full_pose = torch.cat([global_pose, body_pose], dim=1)
        rel_rot_mat = torch.eye(
            3, device=device,
            dtype=dtype).unsqueeze_(dim=0).repeat(batch_size, 1, 1)
        for idx in kin_chain[1:]:
            rel_rot_mat = torch.bmm(full_pose[:, idx], rel_rot_mat)

        # This contains the absolute pose of the parent
        abs_parent_pose = rel_rot_mat.detach()
        # Let's assume that in the input this specific joint is predicted as an absolute value
        abs_joint_pose = body_pose[:, kin_chain[0] - 1]
        # abs_head = parents(abs_neck) * rel_head ==> rel_head = abs_neck.T * abs_head
        rel_joint_pose = torch.matmul(
            abs_parent_pose.reshape(-1, 3, 3).transpose(1, 2),
            abs_joint_pose.reshape(-1, 3, 3))
        # Replace the new relative pose
        body_pose[:, kin_chain[0] - 1, :, :] = rel_joint_pose
        return body_pose
    
    def forward(self, 
                param_dict:dict,
                pose_type='rotmat',
                proj_type='persp'):
        """
            Decode model parameters to smplx vertices & joints & texture
            Args:
                param_dict: smplx parameters
                which may contains:
                    shape_params: [N, number of shape parameters]
                    expression_params: [N, number of expression parameters]
                    global_pose: pelvis pose, [N, 1, 3, 3]
                    body_pose: [N, 21, 3, 3]
                    jaw_pose: [N, 1, 3, 3]
                    eye_pose: [N, 2, 3, 3]
                    left_hand_pose: [N, 15, 3, 3]
                    right_hand_pose: [N, 15, 3, 3]
                pose_type: matrot / aa for matrix rotations or axis angles
            Returns:
                predictions: smplx predictions,
                in which may contains:
                    vertices: [N, number of vertices, 3]
                    landmarks: [N, number of landmarks (68 face keypoints), 3]
                    joints: [N, number of smplx joints (145), 3]
        """
            
        shape_params      = param_dict.get('shape', None)             # torch.Size([1, 250])
        expression_params = param_dict.get('exp', None)               # torch.Size([1, 50])
        global_pose       = param_dict.get('global_pose', None)       # torch.Size([1, 1, 3, 3])
        body_pose         = param_dict.get('body_pose', None)         # torch.Size([1, 21, 3, 3])
        jaw_pose          = param_dict.get('jaw_pose', None)          # torch.Size([1, 1, 3, 3])
        left_hand_pose    = param_dict.get('left_hand_pose', None)    # torch.Size([1, 15, 3, 3])
        right_hand_pose   = param_dict.get('right_hand_pose', None)   # torch.Size([1, 15, 3, 3])
        eye_pose          = param_dict.get('eye_pose', None)          # torch.Size([1, 2, 3, 3])
        eyelid_params     = param_dict.get('eyelid_params', None)     # torch.Size([1, 2])

        if shape_params is None:
            batch_size = global_pose.shape[0]
            shape_params = self.shape_params.expand(batch_size, -1)
        else:
            batch_size = shape_params.shape[0]
        import ipdb ;ipdb.set_trace()
        if proj_type == 'orth':
            projection_func = self.batch_orth_proj
        elif proj_type == 'persp' and param_dict['body_cam'].shape[-1] == 6:
            projection_func = self.batch_persp_proj
        else:
            projection_func = self.batch_week_cam_to_perspective_proj

        if expression_params is None: expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None:
            global_pose = self.global_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': global_pose = torch.zeros_like(global_pose[..., 0])
        if body_pose is None:
            body_pose = self.body_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': body_pose = torch.zeros_like(body_pose[..., 0])
        if jaw_pose is None:
            jaw_pose = self.jaw_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': jaw_pose = torch.zeros_like(jaw_pose[..., 0])
        if eye_pose is None:
            eye_pose = self.eye_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': eye_pose = torch.zeros_like(eye_pose[..., 0])
        if left_hand_pose is None:
            left_hand_pose = self.left_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': left_hand_pose = torch.zeros_like(left_hand_pose[..., 0])
        if right_hand_pose is None:
            right_hand_pose = self.right_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if pose_type == 'aa': right_hand_pose = torch.zeros_like(right_hand_pose[..., 0])
        
        if shape_params.shape[-1] < self.n_shape:
            t_shape_params = torch.cat([shape_params, torch.zeros(shape_params.shape[0], self.n_shape - shape_params.shape[1]).to(shape_params.device)], dim=1)
        else:
            t_shape_params = shape_params[:, :self.n_shape]
        shape_components = torch.cat([t_shape_params, expression_params], dim=1)
        full_pose = torch.cat([global_pose, 
                               body_pose,
                                jaw_pose, 
                                eye_pose,
                                left_hand_pose, 
                                right_hand_pose], dim=1)
        if pose_type == 'rotmat':
            b, n = full_pose.shape[:2]
            full_pose = converter.batch_matrix2axis(full_pose.flatten(0,1)).reshape(b, n, 3)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        # smplx，joints（55x3）
        vertices, joints = lbs(shape_components, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,        # shapedirs[10475, 3, 20]
                          self.J_regressor, self.parents,       # J_regressor([55, 10475])
                          self.lbs_weights, dtype=self.dtype)   # template_vertices（10475x3）
        
        # head_vert = vertices.index_select(1, torch.from_numpy(self.smplx2flame_ind).to(vertices.device))
        head_vert = vertices[:, self.smplx2flame_ind]
        if eyelid_params is not None:
            # coordinates inversed Y issue
            head_vert = head_vert - self.face_r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            head_vert = head_vert - self.face_l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
            vertices[:, self.smplx2flame_ind] = head_vert

        ret_dict = {}
        landmarksmp = vertices2landmarks(head_vert, self.flame_faces_tensor,
                                    self.mp_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                    self.mp_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        ret_dict['face_lmk_mp'] = landmarksmp
        if self.using_lmk203:
            landmarks203 = vertices2landmarks(head_vert, self.flame_faces_tensor,
                                        self.lmk_203_faces_idx.repeat(vertices.shape[0], 1),
                                        self.lmk_203_bary_coords.repeat(vertices.shape[0], 1, 1))
            ret_dict['face_lmk_203'] = landmarks203

        # face dynamic landmarks，lmk_faces_idx（51），lmk_bary_coords（51x3）
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
                find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.head_kin_chain)
            )#dyn_lmk_faces_idx([1, 17])
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,#faces_tensor([20908, 3])
                                       lmk_bary_coords)
        
        final_joint_set = [joints, landmarks]
        if hasattr(self, 'extra_joint_selector'):
            # Add any extra joints that might be needed，extra_joints([1, 22, 3])
            extra_joints = self.extra_joint_selector(vertices, self.faces_tensor)
            final_joint_set.append(extra_joints)
        # Create the final joint set
        joints = torch.cat(final_joint_set, dim=1)
        if self.use_joint_regressor:
            reg_joints = torch.einsum(
                'ji,bik->bjk', self.extra_joint_regressor, vertices)

            # self.source_idxs=self.source_idxs.long()
            # self.source_idxs=self.source_idxs.type(torch.long)#已修改
            # print("self.source_idxs")
            # print(self.source_idxs)

            joints[:, self.source_idxs.long()] = (
                joints[:, self.source_idxs.long()].detach() * 0.0 +
                reg_joints[:, self.target_idxs.long()] * 1.0
            )

        landmarks = torch.cat([landmarks[:, -17:], landmarks[:, :-17]], dim=1)

        # projection
        trans_cam = None
        week_cam = param_dict['body_cam'][..., :3]
        trans_verts = projection_func(vertices, week_cam)
        face_lmks_on_origin_width=landmarks.clone()
        predicted_joints_3d = projection_func(joints, week_cam)
        predicted_joints = predicted_joints_3d[:, :, :2]

        ret_dict['face_lmk_mp'] = projection_func(ret_dict['face_lmk_mp'], week_cam)
        if self.using_lmk203:
            ret_dict['face_lmk_203'] = projection_func(ret_dict['face_lmk_203'], week_cam)
        
        # save predcition
        prediction = {
            # 世界坐标系下3d点
            'vertices': vertices,
            'joints': joints,
            
            # 在弱透视相机投影的2d/3d点
            'transformed_vertices': trans_verts,
            'face_kpt': face_lmks_on_origin_width,
            'smplx_kpt': predicted_joints,
            'smplx_kpt3d': predicted_joints_3d,

            # 相机参数
            'trans_cam': trans_cam,
            'week_cam': week_cam,
            
            # smplx参数            
            'shape': param_dict['shape'],
            'exp': param_dict['exp'],
        }

        ret_dict.update(prediction)
        
        return ret_dict
        
    def pose_rel2abs(self, global_pose, body_pose, abs_joint = 'head'):
        ''' change relative pose to absolute pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        '''
        full_pose = torch.cat([global_pose, body_pose], dim=1)

        if abs_joint == 'head':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == 'neck':
            # Pelvis -> Spine 1, 2, 3 -> Neck -> Head
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == 'right_wrist':
            # Pelvis -> Spine 1, 2, 3 -> right Collar -> right shoulder
            # -> right elbow -> right wrist
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == 'left_wrist':
            # Pelvis -> Spine 1, 2, 3 -> Left Collar -> Left shoulder
            # -> Left elbow -> Left wrist
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(
                f'pose_rel2abs does not support: {abs_joint}')
        rel_rot_mat = torch.eye(3, device=full_pose.device,
                                dtype=full_pose.dtype).unsqueeze_(dim=0)
        for idx in kin_chain:
            rel_rot_mat = torch.matmul(full_pose[:, idx], rel_rot_mat)
        abs_pose = rel_rot_mat[:,None,:,:]
        return abs_pose