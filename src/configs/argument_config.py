import os.path as osp
from dataclasses import dataclass,field
import tyro
from typing_extensions import Annotated
from .base_config import PrintableConfig
from typing import List

@dataclass(repr=False)
class ArgumentConfig(PrintableConfig):
    ########## input arguments ##########
    #source_dir: Annotated[str, tyro.conf.arg(aliases=["-s"])] ='assets/videos'         
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'outputs/test_data/'    # path to driving video or template (.pkl format)

    save_vis_video: bool = False
    tracking_with_interval : bool =False
    save_images: bool = False
    save_visual_render: bool = False
    check_hand_score: float = 0.7
    not_check_hand: bool = False
    
    visible_gpus: Annotated[str, tyro.conf.arg(aliases=["-v"])] = '0,'        # visible gpus, separated by `,`, e.g. 0, 1
    part_lst: Annotated[str, tyro.conf.arg(aliases=["-p"])] = 'nan'           # starts and ends for subprocessing, e.g. 20,40, default: None
    n_divide: Annotated[str, tyro.conf.arg(aliases=["-n"])] = 8               # max divide number
    in_root: Annotated[str, tyro.conf.arg(aliases=["-i"])]  = 'assets/videos' # the input video file paths
    more_in_root: Annotated[List[str], tyro.conf.arg(aliases=["-m"])] = field(default_factory=list)