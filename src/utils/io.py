import os
import yaml
import torch
import pickle
from glob import glob
import os.path as osp
import imageio
import numpy as np
from easydict import EasyDict
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
from PIL import Image

def load_image_rgb(image_path: str):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if image_path.endswith('.mp4'):
        print("read first frame of mp4 file")
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        cap.release()  # 释放资源
        if ret:
            cv2.imwrite('rgb.png', frame)
            img_rgb = Image.open('rgb.png').convert('RGB')
            # 将 BGR 转换为 RGB
            img_rgb = np.array(img_rgb)
            return img_rgb
    if image_path.endswith('.mp4'):
        reader = imageio.get_reader(image_path)
        img = reader.get_data(0)
        return img
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image_rgb(image_path:str, img_rgb:np.ndarray):
    return cv2.imwrite(image_path, cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR))


def load_images_from_directory(directory, n_limit=-1):
    image_paths = sorted(glob(osp.join(directory, '*.png')) + glob(osp.join(directory, '*.jpg')))
    if n_limit > 0: image_paths = image_paths[:n_limit]
    return [load_image_rgb(im_path) for im_path in image_paths]


def load_images_from_video(file_path, n_limit=-1):
    reader = imageio.get_reader(file_path)
    return [image for idx, image in enumerate(reader) if n_limit <= 0 or idx < n_limit]


def load_driving_info(driving_info, n_limit=-1):
    driving_video_ori = []

    if osp.isdir(driving_info):
        driving_video_ori = load_images_from_directory(driving_info, n_limit)
    elif osp.isfile(driving_info):
        driving_video_ori = load_images_from_video(driving_info, n_limit)

    return driving_video_ori


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj


def _resize_to_limit(img: np.ndarray, max_dim=1920, n=2):
    """
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    n = max(n, 1)
    new_h = img.shape[0] - (img.shape[0] % n)
    new_w = img.shape[1] - (img.shape[1] % n)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img


def load_img_online(obj, mode="bgr", **kwargs):
    max_dim = kwargs.get("max_dim", 1920)
    n = kwargs.get("n", 2)
    if isinstance(obj, str):
        if mode.lower() == "gray":
            img = cv2.imread(obj, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(obj, cv2.IMREAD_COLOR)
    else:
        img = obj

    # Resize image to satisfy constraints
    img = _resize_to_limit(img, max_dim=max_dim, n=n)

    if mode.lower() == "bgr":
        return contiguous(img)
    elif mode.lower() == "rgb":
        return contiguous(img[..., ::-1])
    else:
        raise Exception(f"Unknown mode {mode}")


def load_config(config_fp) -> dict:
    with open(config_fp) as fid:
        cfg = yaml.safe_load(fid)
    return EasyDict(cfg)


def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture.
    borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 0], faces[i, 1], faces[i, 2]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
                f.write(s)

                if normal_map is not None:
                    if torch.is_tensor(normal_map):
                        normal_map = normal_map.detach().cpu().numpy().squeeze()

                    normal_map = np.transpose(normal_map, (1, 2, 0))
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')

                    out_normal_map = normal_map / (np.linalg.norm(
                        normal_map, axis=-1, keepdims=True) + 1e-9)
                    out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                    )

            cv2.imwrite(texture_name, texture)


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )


def write_dict_pkl(path, a_dict:dict):
    a_dir = os.path.dirname(path)
    if len(a_dir) > 0: os.makedirs(a_dir, exist_ok=True)
    with open(path, 'wb') as fid:
        pickle.dump(a_dict, fid)


def load_dict_pkl(path, encoding='') -> dict:
    assert os.path.exists(path)
    with open(path, 'rb') as fid:
        if encoding != '':
            ret = pickle.load(fid, encoding=encoding)
        else:
            ret = pickle.load(fid)
    return ret
