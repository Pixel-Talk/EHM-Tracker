import os.path as osp
from dataclasses import dataclass
from typing import Literal, Tuple
from .base_config import PrintableConfig

@dataclass(repr=False)  # use repr from PrintableConfig
class DataPreparationConfig(PrintableConfig):
    mp_cfg_path:      str = 'src/configs/model_configs/mediapipe_detector.yaml'
    teaser_cfg_path:   str = 'src/configs/model_configs/teaser_onnx_config.yaml'
    hamer_cfg_path:   str = 'src/configs/model_configs/hamer_onnx_config.yaml'
    dwpose_cfg_path:  str = 'src/configs/model_configs/dwpose_onnx_config.yaml'
    kp70_cfg_path:    str = 'src/configs/model_configs/lmk70_onnx_config.yaml'
    matting_cfg_path: str = 'src/configs/model_configs/matting_config.yaml'
    pixie_cfg_path:   str = 'src/configs/model_configs/pixie_config.yaml'

    kp203_path:       str = 'pretrained/lmk106/landmark.onnx'
    fan_path:         str = 'pretrained/fan/fan_lmk_detector.onnx'
    flame_assets_dir: str = 'assets/FLAME'
    mano_assets_dir:  str = 'assets/MANO'
    smplx_assets_dir: str = 'assets/SMPLX'
    vposer_ckpt_dir:  str = 'pretrained/vposer/vposer_v1_0'

    projection_type: str = 'persp'      # projection type, in orth, persp
    input_shape:    Tuple[int, int] = (256, 256)  # input shape
    output_format:  Literal['mp4', 'gif'] = 'mp4'  # output video format
    output_fps:     int = 30  # fps for output video
    crf:            int = 15  # crf for output video
    log_interval:   int = 100
    use_vposer:     bool = True

    tanfov:       float = 1/24
    body_crop_size:    int = 224
    teaser_input_size:  int = 224
    head_crop_size:    int = 512
    hand_crop_size:    int = 512
    body_hd_size:      int = 1024
    device_id:         int = 0
    device :           str = f'cuda:{device_id}'
    flag_do_crop:      bool = False  # whether to crop the reference portrait to the face-cropping space
    flag_do_rot:       bool = True  # whether to conduct the rotation when flag_do_crop is True
    with_optimization: bool = True

    fit_flame: bool = True
    fit_ehm:   bool = True
    check_skip_extraction: bool = True
    
    tracking_with_interval: bool = False
    save_images: bool = False
    save_visual_render: bool = False
    default_frame_interval: int = 6
    max_frames: int = 75
    min_frames: int = 5
    
    check_hand_score: float = 0.7
    check_hand_dist:  float = body_hd_size*3
    not_check_hand: bool = False
    
