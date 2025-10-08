import os.path as osp
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
import onnxruntime
from .timer import Timer
from .rprint import rlog
from .crop import crop_image, _transform_pts
from numba import jit


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def to_ndarray(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.array(obj)


@jit(nopython=True)
def t_get_preds_fromhm(hm, idx, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps and the
    coresponding locations of the maximums. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    idx += 1
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
    preds[:, :, 1] = np.floor((preds[:, :, 1] - 1) / H) + 1

    for i in range(B):
        for j in range(C):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = np.array(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                    hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j] += np.sign(diff) * 0.25

    preds -= 0.5

    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(preds[i, j], center, scale, H, True)

    return preds, preds_orig


@jit(nopython=True)
def transform_np(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point


class FanLandmarkRunner(object):
    """landmark runner"""
    def __init__(self, **kwargs):
        ckpt_path = kwargs.get('ckpt_path')
        onnx_provider = kwargs.get('onnx_provider', 'cuda')  # 默认用cuda
        device_id = kwargs.get('device_id', 0)
        self.dsize = kwargs.get('dsize', 256)
        self.timer = Timer()

        if onnx_provider.lower() == 'cuda':
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=[
                    ('CUDAExecutionProvider', {'device_id': device_id})
                ]
            )
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 4  # 默认线程数为 4
            self.session = onnxruntime.InferenceSession(
                ckpt_path, providers=['CPUExecutionProvider'],
                sess_options=opts
            )

    def get_preds_fromhm(self, hm, center=None, scale=None):
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        B, C, H, W = hm.shape
        hm_reshape = hm.reshape(B, C, H * W)
        idx = np.argmax(hm_reshape, axis=-1)
        scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)
        preds, preds_orig = t_get_preds_fromhm(hm, idx, center, scale)

        return preds, preds_orig, scores

    def _run(self, inp):
        out = self.session.run(None, {'input_image': inp})
        return out

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

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_hm = self._run(inp)[0]

        center = np.array([self.dsize / 2, self.dsize / 2])
        center[1] = center[1] - self.dsize * 0.12
        
        pts, pts_img, scores = self.get_preds_fromhm(out_hm, center, 1.)

        pts_img = pts_img.reshape((68, 2))
        scores = np.squeeze(scores, 0)

        pts = pts_img   # self.dsize  # scale to 0-224
        pts = _transform_pts(pts, M=crop_dct['M_c2o'])

        return {
            'pts': pts,  # 2d landmarks 68 points
            'scores': scores
        }

    def warmup(self):
        # 构造dummy image进行warmup
        self.timer.tic()

        dummy_image = np.zeros((1, 3, self.dsize, self.dsize), dtype=np.float32)

        _ = self._run(dummy_image)

        elapse = self.timer.toc()
        rlog(f'FanLandmarkRunner warmup time: {elapse:.3f}s')
