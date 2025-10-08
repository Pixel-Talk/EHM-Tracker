"""
utility functions and classes to handle feature extraction and model loading
"""

import os
import cv2
import torch
import importlib
import numpy as np
import os.path as osp
from .rprint import rlog as log
from collections import OrderedDict
from ..modules.base.onnx_model import OnnxModel


def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def insert_dict(adict:dict, key, elem:object,):
    if key not in adict: adict[key] = []
    adict[key].append(elem)
    return adict


def create_multi_dirs(*dir_lst):
    for a_dir in dir_lst:
        os.makedirs(a_dir, exist_ok=True)


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def get_recurrent_index(i: int, length: int, generated_lenth: int=-1, ralign: bool=False):
    """get recurrent index. [0, 1, 2, 2, 1, 0, 0, 1, 2, ....]
    For example: 
        i = 6, length=3, return 0;
        i = 6, length=7, return 6;
        i = 6, length=4, return 1

    Args:
        i (int): index begins with 0
        length (int): total frames
        generated_length (int): generated video length
        ralign: if align frames to right, defaults to False
    """
    if ralign:
        i = generated_lenth - i - 1
    if i // length % 2 == 0:
        idx = i % length
    else:
        idx = length - i % length - 1
    
    if ralign:
        return length - idx - 1
    return idx


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            log(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2cuda(dct: dict, device_id: int = 0):
    for key in dct:
        dct[key] = torch.tensor(dct[key]).cuda(device_id)
    return dct


def dct2numpy(dct: dict):
    for key in dct:
        if isinstance(dct[key], (torch.Tensor, torch.FloatTensor)):
            dct[key] = dct[key].detach().cpu().numpy()
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def get_obj_from_str(string, reload=False) -> object:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    obj = config.pop('target')
    return get_obj_from_str(obj)(**config['params'])


def load_model(model_config, device) -> torch.nn.Module:
    model = instantiate_from_config(model_config)
    model.eval()

    dev = torch.device(device)

    if not "ckpt_fp" in model_config:
        log("No `ckpt_fp` to found in `model_config`, init it from scratch.")
        model.to(dev)
        return model
    
    ckpt_fp = model_config.pop('ckpt_fp')
    assert ckpt_fp

    state_dict = torch.load(ckpt_fp, map_location="cpu")
    state_dict = remove_ddp_dumplicate_key(state_dict)
    x, y = model.load_state_dict(state_dict, strict=False)
    if len(y) > 0: log(f'Unexpected keys: {y}')
    # if len(x) > 0: log(f'Missing keys: {x}')

    model.to(dev)
    return model


def load_onnx_model(model_config, device=None) -> OnnxModel:
    if device is not None:
        model_config.is_gpu = 'cuda' in str(device)
    
    model = instantiate_from_config(model_config)
    return model


def assign_attributes(a_from, a_to):
    _a_from = a_from 
    if type(a_from) is not dict:
        _a_from = vars(a_from)
    for k, v in _a_from.items():
        if isinstance(a_to, dict):
            if k in a_to:
                a_to[k] = v
        else:
            if hasattr(a_to, k):
                setattr(a_to, k, v)


# get coefficients of Eqn. 7
def calculate_transformation(config, s_kp_info, t_0_kp_info, t_i_kp_info, R_s, R_t_0, R_t_i):
    if config.relative:
        new_rotation = (R_t_i @ R_t_0.permute(0, 2, 1)) @ R_s
        new_expression = s_kp_info['exp'] + (t_i_kp_info['exp'] - t_0_kp_info['exp'])
    else:
        new_rotation = R_t_i
        new_expression = t_i_kp_info['exp']
    new_translation = s_kp_info['t'] + (t_i_kp_info['t'] - t_0_kp_info['t'])
    new_translation[..., 2].fill_(0)  # Keep the z-axis unchanged
    new_scale = s_kp_info['scale'] * (t_i_kp_info['scale'] / t_0_kp_info['scale'])
    return new_rotation, new_expression, new_translation, new_scale


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def resize_to_limit(img, max_dim=1280, n=2):
    h, w = img.shape[:2]
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))
    n = max(n, 1)
    new_h = img.shape[0] - (img.shape[0] % n)
    new_w = img.shape[1] - (img.shape[1] % n)
    if new_h == 0 or new_w == 0:
        return img
    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]
    return img


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)
    return image.astype(np.uint8)


def image2tensor(image, norm=True):
    tensor = image.transpose(2,0,1)
    if norm: tensor = tensor / 255.
    tensor = torch.from_numpy(tensor).float()
    return tensor


def mount_model(model):
    model.cuda()


def unmount_model(model):
    model.cpu()
    torch.cuda.empty_cache()


def face_vertices(vertices, faces):
    """
    borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def get_gpu_info():
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count= pynvml.nvmlDeviceGetCount()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name=pynvml.nvmlDeviceGetName(handle)
        gpu_version=pynvml.nvmlSystemGetDriverVersion()
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        B_to_MB=1024*1024
        gpu_Total = info.total/B_to_MB
        gpu_Free = info.free /B_to_MB
        gpu_Used = info.used /B_to_MB
        
        return dict(
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            gpu_version=gpu_version,
            gpu_Total=gpu_Total,
            gpu_Free=gpu_Free,
            gpu_Used=gpu_Used
        )
    except:
        import traceback
        traceback.print_exc()
        

def get_machine_info():
    import platform

    host_name=platform.node()
    gpu_info = get_gpu_info()
    
    machine_info=dict(
        host_name=host_name,
        gpu_info=gpu_info,
    )
    return machine_info


def build_minibatch(all_frames, batch_size=1024, share_id=False):
    if share_id:
        all_frames = sorted(all_frames)
        video_names = list(set(['_'.join(frame_name.split('_')[:-1]) for frame_name in all_frames]))
        video_frames = {video_name: [] for video_name in video_names}
        for frame in all_frames:
            video_name = '_'.join(frame.split('_')[:-1])
            video_frames[video_name].append(frame)
        all_mini_batch = []
        for video_name in video_names:
            mini_batch = []
            for frame_name in video_frames[video_name]:
                mini_batch.append(frame_name)
                if len(mini_batch) % batch_size == 0:
                    all_mini_batch.append(mini_batch)
                    mini_batch = []
            if len(mini_batch):
                all_mini_batch.append(mini_batch)
    else:
        try:
            all_frames = sorted(all_frames, key=lambda x: int(x.split('_')[-1]))
        except:
            all_frames = sorted(all_frames)
        all_mini_batch, mini_batch = [], []
        for frame_name in all_frames:
            mini_batch.append(frame_name)
            if len(mini_batch) % batch_size == 0:
                all_mini_batch.append(mini_batch)
                mini_batch = []
        if len(mini_batch):
            all_mini_batch.append(mini_batch)
    return all_mini_batch


