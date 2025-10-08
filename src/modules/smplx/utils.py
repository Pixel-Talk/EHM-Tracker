import numpy as np
import torch
import roma
from ...utils import rotation_converter as converter

def convert_pose(param_dict, flip_left_hand=True):
    ''' Convert pose parameters to rotation matrix
    Args:
        param_dict: smplx parameters
        param_type: should be one of body/head/hand
    Returns:
        param_dict: smplx parameters 
    '''
    # convert pose representations: the output from network are continous repre or axis angle,
    # while the input pose for smplx need to be rotation matrix
    for key in param_dict:
        if "pose" in key and 'jaw' not in key:
            if len(param_dict[key].shape) == 4 and param_dict[key].shape[-1] == 3 and param_dict[key].shape[-2] == 3:
                continue
            param_dict[key] = converter.batch_cont2matrix(param_dict[key])
    
    param_dict['jaw_pose'] = converter.batch_euler2matrix(param_dict['jaw_pose'])[:, None, :, :]

    # the predcition from the head and hand share regressor is always absolute pose
    param_dict['abs_head_pose'] = param_dict['head_pose'].clone()
    param_dict['abs_right_wrist_pose'] = param_dict['right_wrist_pose'].clone()
    param_dict['abs_left_wrist_pose'] = param_dict['left_wrist_pose'].clone()
    if flip_left_hand:
        # the body-hand share regressor is working for right hand
        # so we assume body network get the flipped feature for the left hand. then get the parameters
        # then we need to flip it back to left, which matches the input left hand
        param_dict['left_wrist_pose'] = converter.flip_pose(param_dict['left_wrist_pose']).clone()
        param_dict['left_hand_pose'] = converter.flip_pose(param_dict['left_hand_pose']).clone()

        if 'hand_l__hand_pose' in param_dict.keys():
            param_dict['hand_l__hand_pose'] = converter.flip_pose(param_dict['hand_l__hand_pose']).clone()

    return param_dict


def pose_abs2rel(global_pose, body_pose, abs_joint = 'head'):
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


def orginaze_body_pose(coeff_lst_dct:dict, is_numpy=False):
    if is_numpy:
        coeff_lst_dct = {k: torch.from_numpy(v) for k, v in coeff_lst_dct.items()}
    # concatenate body pose
    if 'jaw_pose' in coeff_lst_dct.keys() and len(coeff_lst_dct['jaw_pose'].shape) == 2:
        convert_pose(coeff_lst_dct, flip_left_hand=True)
    elif coeff_lst_dct['right_wrist_pose'].shape[-1] == 6:
        convert_pose(coeff_lst_dct, flip_left_hand=True)

    partbody_pose = coeff_lst_dct['partbody_pose']
    coeff_lst_dct['body_pose'] = torch.cat(
        [partbody_pose[:, :11],
            coeff_lst_dct['neck_pose'],
            partbody_pose[:, 11:11+2],
            coeff_lst_dct['head_pose'],
            partbody_pose[:, 13:13+4],
            coeff_lst_dct['left_wrist_pose'],
            coeff_lst_dct['right_wrist_pose']], dim=1)
    
    # change absolute head&hand pose to relative pose according to rest body pose
    coeff_lst_dct['body_pose'] = pose_abs2rel(
        coeff_lst_dct['global_pose'],
        coeff_lst_dct['body_pose'],
        abs_joint='head')
    coeff_lst_dct['body_pose'] = pose_abs2rel(
        coeff_lst_dct['global_pose'],
        coeff_lst_dct['body_pose'],
        abs_joint='left_wrist')
    coeff_lst_dct['body_pose'] = pose_abs2rel(
        coeff_lst_dct['global_pose'],
        coeff_lst_dct['body_pose'],
        abs_joint='right_wrist')
    
    # check if pose is natural (relative rotation), if not, set relative to 0 (especially for head pose)
    # xyz: pitch(positive for looking down), yaw(positive for looking left), roll(rolling chin to left)
    for pose_ind in [14]:  # head [15-1, 20-1, 21-1]:
        curr_pose = coeff_lst_dct['body_pose'][:, pose_ind]
        euler_pose = converter._compute_euler_from_matrix(curr_pose)
        for i, max_angle in enumerate([20, 70, 10]):
            euler_pose_curr = euler_pose[:, i]
            euler_pose_curr[
                euler_pose_curr !=
                torch.clamp(
                    euler_pose_curr,
                    min=-max_angle*np.pi/180,
                    max=max_angle*np.pi/180)
            ] = 0.
        coeff_lst_dct['body_pose'][:, pose_ind] = converter.batch_euler2matrix(euler_pose)

    ret_dict = {}
    del coeff_lst_dct['partbody_pose']
    for k, v in coeff_lst_dct.items():
        if 'pose' in k:
            b, n = v.shape[:2]
            ret_dict[k] = converter.batch_matrix2axis(v.flatten(0, 1)).reshape(b, n, 3)
        else:
            ret_dict[k] = v
    
    if is_numpy:
        for k, v in ret_dict.items():
            ret_dict[k] = v.numpy()
    
    return ret_dict

def rotate_global_pose(coeff_param):
    axi_angle=coeff_param["global_pose"]
    rot_mat=roma.rotvec_to_rotmat(axi_angle[0])
    rot_mat=torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]],dtype=torch.float32,device=rot_mat.device)@rot_mat
    axi_angle=roma.rotmat_to_rotvec(rot_mat)[None]
    coeff_param["global_pose"]=axi_angle
    return coeff_param