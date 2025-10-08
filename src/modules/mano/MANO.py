import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import os.path as osp
from typing import NewType, Optional, Union
from ...utils.rprint import rlog as log
from ..smplx.lbs import vertices2joints, lbs
from ...utils import rotation_converter as converter

Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_tensor(
        array: Union[Array, Tensor], dtype=torch.float32
) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)
    

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class VertexJointSelector(nn.Module):

    def __init__(self, vertex_ids=None, **kwargs):
        super(VertexJointSelector, self).__init__()

        self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        tips_idxs = []
        for tip_name in self.tip_names:
            tips_idxs.append(vertex_ids[tip_name])

        extra_joints_idxs = np.array(tips_idxs)

        self.register_buffer('extra_joints_idxs', to_tensor(extra_joints_idxs, dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints


class MANO(nn.Module):
    # The hand joints are replaced by MANO
    NUM_BODY_JOINTS = 1
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS

    def __init__(self, mano_assets_dir:str, use_pca: bool = True,
                 num_pca_comps: int = 6, flat_hand_mean: bool = False, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        """
        super(MANO, self).__init__()

        with open(osp.join(mano_assets_dir, 'MANO_RIGHT.pkl'), 'rb') as fid:
            model_data = pickle.load(fid, encoding='latin1')
        data_struct = Struct(**model_data)

        mano_vert_ids = {
            'thumb': 744,
            'index': 320,
            'middle': 443,
            'ring':	 554,
            'pinky': 671,
        }

        self.dtype = torch.float32
        self.use_pca = use_pca
        self.num_pca_comps = num_pca_comps
        if self.num_pca_comps == 45:
            self.use_pca = False
        self.flat_hand_mean = flat_hand_mean

        log('Using generic MANO model')
 
        hand_components = data_struct.hands_components[:num_pca_comps]

        self.np_hand_components = hand_components

        self.faces = data_struct.f
        self.register_buffer('faces_tensor', to_tensor(to_np(self.faces, dtype=np.int64),  dtype=torch.long))

        if self.use_pca:
            self.register_buffer('hand_components', torch.tensor(hand_components, dtype=torch.float32))

        if self.flat_hand_mean:
            hand_mean = np.zeros_like(data_struct.hands_mean)
        else:
            hand_mean = data_struct.hands_mean

        self.register_buffer('hand_mean', to_tensor(hand_mean, dtype=self.dtype))

        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data_struct.posedirs.shape[-1]
        # 207 x 20670
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs',    to_tensor(to_np(posedirs), dtype=self.dtype))

        self.register_buffer('shapedirs',   to_tensor(to_np(data_struct.shapedirs), dtype=self.dtype))
        self.register_buffer('lbs_weights', to_tensor(to_np(data_struct.weights), dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(data_struct.J_regressor), dtype=self.dtype))

        self.vertex_joint_selector = VertexJointSelector(vertex_ids=mano_vert_ids, **kwargs)

        # add only MANO tips to the extra joints
        self.vertex_joint_selector.extra_joints_idxs = to_tensor(
            list(mano_vert_ids.values()), dtype=torch.long)

        betas = torch.zeros([1, 10], dtype=self.dtype)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))

        self.register_buffer('v_template', to_tensor(to_np(data_struct.v_template), dtype=self.dtype))

        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        self.register_buffer('extra_joints_idxs', to_tensor(list(mano_vert_ids.values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))
        self.register_buffer('global_orient', torch.eye(3, dtype=self.dtype).view(1, 1, 3, 3))

        self.register_buffer('hand_pose', torch.eye(3, dtype=self.dtype).view(1, 1, 3, 3).expand(1, 15, -1, -1).contiguous())
        self.register_buffer('transl',    torch.zeros([1, 3], dtype=self.dtype))

        self.selected_vert_ids = np.load(os.path.join(mano_assets_dir, 'selected_hand_ver.npy'))

    def forward(self, param_dictionary: dict, pose_type='rotmat'):
        
        betas = param_dictionary.get('betas', self.betas)
        global_orient = param_dictionary.get('global_orient', self.global_orient)
        hand_pose = param_dictionary.get('hand_pose', self.hand_pose)
        apply_trans = 'transl' in param_dictionary.keys()
        transl = param_dictionary.get('transl', self.transl)

        if len(global_orient.shape) != 3: global_orient = global_orient.squeeze().unsqueeze(1)

        if global_orient.shape[-2] == 3 and global_orient.shape[-1] == 3:
            global_orient = converter.batch_matrix2axis(global_orient).unsqueeze(1)
        
        if hand_pose.shape[-2] == 3 and hand_pose.shape[-1] == 3:
            b, n = hand_pose.shape[:2]
            hand_pose = converter.batch_matrix2axis(hand_pose.flatten(0,1)).reshape(b, n, 3)

        full_pose = torch.cat([global_orient, hand_pose], dim=1)

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=pose_type=='aa')
        
        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, vertices)
            joints = torch.cat([joints, extra_joints], dim=1)

        return dict(vertices=vertices, joints=joints)
