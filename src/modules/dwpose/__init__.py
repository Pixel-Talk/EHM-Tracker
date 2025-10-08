# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody
from ...utils.timer import Timer
from .onnxdet import inference_detector
from ...utils.rprint import rlog as log


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, ckpt_dir:str, use_gpu=True):
        self.pose_estimation = Wholebody(ckpt_dir, use_gpu=use_gpu)
        self.timer = Timer()

    def warmup(self):
        self.timer.tic()

        self(np.zeros((512, 512, 3), np.uint8))

        elapse = self.timer.toc()
        log(f'DWposeDetector warmup time: {elapse:.3f}s')
    
    def get_max_area_bbox_index(self, keypoints):
        x_min = keypoints[:, :, 0].min(axis=1)  
        y_min = keypoints[:, :, 1].min(axis=1)  
        x_max = keypoints[:, :, 0].max(axis=1)  
        y_max = keypoints[:, :, 1].max(axis=1)  

        widths = x_max - x_min  # x_max - x_min
        heights = y_max - y_min  # y_max - y_min
        
        areas = widths * heights
        
        max_area_index = np.argmax(areas)
        
        return max_area_index

    def __call__(self, oriImg:np.ndarray):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset, bbox = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape

            if subset.shape[0] > 1:
                selcted_idx = self.get_max_area_bbox_index(candidate)
                candidate = candidate[selcted_idx:selcted_idx+1]
                subset = subset[selcted_idx:selcted_idx+1]
                bbox = bbox[selcted_idx:selcted_idx+1]

            body = candidate[:,:18].copy()
            body = body.reshape(18, locs)

            raw_pose = dict(keypoints=candidate.squeeze(), scores=subset.squeeze(), bbox=bbox.squeeze())

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            score = subset[:,:18].copy()
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1
            
            bodies = dict(candidate=body.squeeze(), subset=score.squeeze())
            pose = dict(bodies=bodies, hands=hands.squeeze(), faces=faces.squeeze(), feet=foot.squeeze(), bbox=bbox.squeeze())

        return pose, raw_pose