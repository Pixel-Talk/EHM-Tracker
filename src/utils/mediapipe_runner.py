import math
import os.path as osp
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .timer import Timer
from .rprint import rlog
from .crop import crop_image, _transform_pts


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


class MediapipeRunner(object):
    """Mediapipe face landmarks detection runner"""
    def __init__(self, ckpt_fp='pretrained/mediapipe/face_landmarker.task', dsize=224):
        base_options = python.BaseOptions(model_asset_path=ckpt_fp)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1,
                                            min_face_detection_confidence=0.1,
                                            min_face_presence_confidence=0.1
                                            )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.dsize  = dsize
        self.timer  = Timer()

        self.lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 
                            76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 307, 320, 404, 315, 16, 85, 180, 90, 77,
                            62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96,
                            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    def _run(self, img_rgb):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = self.detector.detect(image)

        if len(results.face_landmarks) == 0:
            rlog('No face detected')
            return None
        
        face_landmarks_numpy = np.zeros((478, 3))

        face_landmarks = results.face_landmarks[0]
        for i, landmark in enumerate(face_landmarks):
            face_landmarks_numpy[i] = [landmark.x*image.width, landmark.y*image.height, landmark.z]

        out_pts = face_landmarks_numpy[..., :2]

        return out_pts

    def run(self, img_rgb: np.ndarray, lmk=None):
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct['img_crop']
        else:
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        out_pts = self._run(img_crop_rgb)

        if out_pts is None:
            return {'pts': out_pts}

        pts = _transform_pts(out_pts, M=crop_dct['M_c2o'])

        return {
            'pts': pts,  # 2d landmarks 478 points
        }

    def warmup(self):
        # 构造dummy image进行warmup
        self.timer.tic()

        dummy_image = np.zeros((self.dsize, self.dsize, 3), dtype=np.uint8)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        rlog(f'MediapipeRunner warmup time: {elapse:.3f}s')
