import cv2
import numpy as np
import mediapipe as mp
from typing import Mapping
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC


def number_to_rgb(n):
    n = int(n)
    r = (n >> 16) & 0xFF
    g = (n >> 8) & 0xFF
    b = n & 0xFF
    return (r, g, b)


def draw_landmarks(landmarks, image, color=(0, 255, 0), radius=2, thickness=-1, viz_index=False):
    ret_img = image.copy()
    for (idx, lmk) in enumerate(landmarks):
        if len(lmk) > 2:
            color = number_to_rgb(lmk[2]*1500)
        ret_img = cv2.circle(ret_img, (int(lmk[0]), int(lmk[1])), radius, color, thickness)
        if viz_index:
            cv2.putText(ret_img, f'{idx}', (int(lmk[0]), int(lmk[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, ((idx*2)%255, 145, (idx*3)%255), 1)
    return ret_img


def _draw_mp_kps(image, landmarks, connections, landmark_drawing_spec, connection_drawing_spec):
    t_landmarks = np.int32(landmarks[:, :2])
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        drawing_spec = connection_drawing_spec[connection] if isinstance(
            connection_drawing_spec, Mapping) else connection_drawing_spec
        cv2.line(image, t_landmarks[start_idx],
                 t_landmarks[end_idx], drawing_spec.color,
                 drawing_spec.thickness)
    
    if landmark_drawing_spec:
        for idx in range(t_landmarks.shape[0]):
            landmark_px = t_landmarks[idx]
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
            landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                        int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, (224, 224, 224),
                        drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                        drawing_spec.color, drawing_spec.thickness)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_connections = mp.solutions.face_mesh_connections

def draw_mediapipe_kps(mp_kps, canvas):
    ret_img = canvas.copy()
    # draw FACEMESH_TESSELATION
    _draw_mp_kps(ret_img, mp_kps, 
        mp.solutions.face_mesh.FACEMESH_TESSELATION, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_CONTOURS
    _draw_mp_kps(ret_img, mp_kps, 
        mp.solutions.face_mesh.FACEMESH_CONTOURS, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(), 
        landmark_drawing_spec=None)

    # draw FACEMESH_IRISES
    _draw_mp_kps(ret_img, mp_kps, 
        mp.solutions.face_mesh.FACEMESH_IRISES, 
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(), 
        landmark_drawing_spec=None)
    
    return ret_img


def get_semantic_indices():

    semantic_connections = {
        'Contours':     mp_connections.FACEMESH_CONTOURS,
        'FaceOval':     mp_connections.FACEMESH_FACE_OVAL,
        'LeftIris':     mp_connections.FACEMESH_LEFT_IRIS,
        'LeftEye':      mp_connections.FACEMESH_LEFT_EYE,
        'LeftEyebrow':  mp_connections.FACEMESH_LEFT_EYEBROW,
        'RightIris':    mp_connections.FACEMESH_RIGHT_IRIS,
        'RightEye':     mp_connections.FACEMESH_RIGHT_EYE,
        'RightEyebrow': mp_connections.FACEMESH_RIGHT_EYEBROW,
        'Lips':         mp_connections.FACEMESH_LIPS,
        'Tesselation':  mp_connections.FACEMESH_TESSELATION
    }

    def get_compact_idx(connections):
        ret = []
        for conn in connections:
            ret.append(conn[0])
            ret.append(conn[1])
        
        return sorted(tuple(set(ret)))
    
    semantic_indexes = {k: get_compact_idx(v) for k, v in semantic_connections.items()}

    return semantic_indexes


mp_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 
                  76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306, 307, 320, 404, 315, 16, 85, 180, 90, 77,
                  62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96,
                  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
mp_lowerface_indices = [177, 132, 215, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 435, 361, 401,
                        147, 213, 192, 138, 135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367, 433, 416, 376,
                        57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 
                        212, 216, 206, 203, 423, 426, 436, 432, 422, 424, 418, 421, 200, 201, 194, 204, 202, 
                        214, 207, 205, 425, 427, 434, 430, 431, 262, 428, 199, 208, 32, 211, 210]     # todo: need further check indices
chain_indices = [93, 137, 123, 50, 205, 206, 165, 167, 164, 393, 391, 426, 425, 280, 352, 366, 323] 
teeth_indices = [62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292, 325, 319, 403, 316, 15, 86, 179, 89, 96]
nose_indices  = [142, 129, 98, 97, 2, 326, 327, 358, 371, 355, 437, 399, 419, 197, 196, 174, 217, 126,
                  102, 64, 240, 99, 60, 75, 59, 235, 166, 219, 48, 49, 209,
                  331, 294, 460, 328, 290, 305, 289, 392, 439, 278, 279, 360,
                  198, 131, 115, 218, 79, 20, 242, 141, 94, 370, 462, 250, 309, 438, 344, 360, 420,
                  236, 3, 195, 248, 456, 363, 440, 457, 459, 458, 461, 354, 19, 125, 241, 238, 239, 237, 220, 134,
                  51, 5, 281, 275, 274, 354, 19, 125, 44, 45, 1, 4]
kp68_lowerface_indices = [x for x in range(3, 14)] + [52, 50, 3]
kp68_mouth_indices = [x for x in range(48, 61)]


def _merge_with_weight(p1, p2, w):
    return p1 * (1 - w) + p2 * w


def merge_lower_face_pints(pred_lst, pred_ff_lst, ff_only=False):
    if ff_only is True:
        w1, w2 = 0.95, 0.95
    else:
        w1, w2 = 0.8, 0.2
    pred_lst[mp_lip_indices] = _merge_with_weight(pred_lst[mp_lip_indices], pred_ff_lst[:len(mp_lip_indices)], w1)
    pred_lst[mp_lowerface_indices] = _merge_with_weight(pred_lst[mp_lowerface_indices], pred_ff_lst[len(mp_lip_indices):], w2)
    return pred_lst


def mask_from_points(size, points, radius=0, is_converx=True, mean_y=-1, ratio=0.025):
    """ Create a mask of supplied size from supplied points
    :param size: tuple of output mask size
    :param points: array of [x, y] points
    :returns: mask of values 0 and 255 where
                255 indicates the convex hull containing the points
    """
    if radius == 0:
        radius = int(size[0] * ratio)
    kernel = np.ones((int(abs(radius)), int(abs(radius))), np.uint8)

    if mean_y > 0:
        v_ids = points[:, :, 1] > mean_y
        points[v_ids, 1] = mean_y + (points[v_ids, 1] - mean_y) * 1.3

    mask = np.zeros(size, np.uint8)
    if is_converx:
        p = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, np.squeeze(p).astype(np.int32), 255)
    else:
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)

    if radius < 0:
        mask = cv2.erode(mask, kernel)
    else:
        mask = cv2.dilate(mask, kernel)

    return mask


def draw_teeth_mask(ver, canvas):
    teeth_mask = mask_from_points(canvas.shape[:2], ver[:, teeth_indices, ...], radius=3)
    teeth_mask = cv2.merge([np.zeros_like(teeth_mask), teeth_mask, teeth_mask])
    canvas = np.clip(canvas / 1.0 + teeth_mask / 2.0, 0, 225)
    return canvas.astype(np.uint8)


def draw_nose_mask(size, ver):
    nose_mask = mask_from_points(size, ver[:, nose_indices, ...])
    return nose_mask


def draw_lowerface_mask(size, ver, mean_y, is_kp68=False):
    if not is_kp68:
        lf_idx = mp_lowerface_indices + mp_lip_indices + chain_indices
        lf_mask = mask_from_points(size, ver[:, lf_idx, ...], mean_y=mean_y)
    else:

        lf_idx = kp68_lowerface_indices
        lf_mask = mask_from_points(size, ver[:, lf_idx, ...], is_converx=False, mean_y=mean_y)
    return lf_mask


def draw_mouth_mask(size, ver, is_kp68=False):
    if not is_kp68:
        lf_idx = mp_lip_indices
        lf_mask = mask_from_points(size, ver[:, lf_idx, ...])
    else:
        lf_idx = kp68_mouth_indices
        lf_mask = mask_from_points(size, ver[:, lf_idx, ...], radius=27, is_converx=False)
    return lf_mask


def draw_flame_lowerface_condition(img_fg:np.ndarray, img_fg_alpha:np.ndarray, img_bg:np.ndarray, lmk203:np.ndarray):
    """draw flame lower face condition

    Args:
        img_fg (np.ndarray): foreground image in 0~1, HxWxC
        img_fg_alpha (np.ndarray): foreground image alpha in 0~1, HxW
        img_bg (np.ndarray): background image in 0~1, HxWxC
        lmk203 (np.ndarray): landmark in 203x2
    """
    ## 1. draw lower mask
    lower_pts = np.concatenate([lmk203[114: 139], ((lmk203[57] + lmk203[202])/2)[None, ...]], axis=0)
    # # enlarge lower y pts
    lower_pts[:, 1] = (lower_pts[:, 1] - lmk203[201, 1]) * 1.2 + lmk203[201, 1]
    mask1 = mask_from_points(img_bg.shape[:2], lower_pts, ratio=0.03, is_converx=False) / 255   # todo: check this and regenerate condition images
    mouth_region = mask_from_points(img_bg.shape[:2], lmk203[84: 108], ratio=0.015, is_converx=False) / 255
    mask1[mouth_region>0.05] = img_fg_alpha[mouth_region>0.05]

    final_mask = np.clip(img_fg_alpha + mask1, 0, 1)
    final_mask = cv2.merge([final_mask]*3)

    ## 2. draw inner mouth region
    bg_img = img_bg * 255
    bg_img[mouth_region>0.05] = (225, 225, 0)

    ## 3. overlay them
    ret_img = final_mask * img_fg + (1 - final_mask) * bg_img / 255

    return np.clip(ret_img, 0, 1)


def draw_fullface_mask(size, ver, mean_y):
    lf_idx = mp_lowerface_indices + mp_lip_indices + chain_indices
    lf_mask = mask_from_points(size, ver[:, lf_idx, ...], mean_y=mean_y)
    ff_mask = mask_from_points(size, ver)
    ff_mask = np.uint8(((lf_mask / 1.0 + ff_mask / 1.0) > 0) * 255.)
    return ff_mask


def get_bbox_from_vert(vert):
    rect = [vert[:,0].min(), vert[:,1].min(), vert[:,0].max(), vert[:,1].max()]

    cx = rect[0] + (rect[2] - rect[0]) / 2
    cy = rect[1] + (rect[3] - rect[1]) / 2

    size = max(rect[2]- rect[0], rect[3]- rect[1])
    size = size * 1.3

    return [int(cx - size/2), int(cy-size/2), int(cx + size/2), int(cy + size/2)]


def alpha_feathering(src_img, dest_img, img_mask, use_blur=True):
    if use_blur:
        blur_radius = int(src_img.shape[0]/30)
        mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    else:
        mask = img_mask
    mask = mask / 255.0
    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1-mask)

    return result_img

