import torch
import pickle
import os.path as osp
import numpy as np
from ...utils.graphics import GS_BaseMeshRenderer

class Renderer(GS_BaseMeshRenderer):
    def __init__(self, assets_dir, image_size=512, device='cuda', focal_length=24):
        super().__init__( image_size, focal_length=focal_length)
        self.focal_length=focal_length
        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        model_data = pickle.load(open(osp.join(assets_dir, 'MANO_RIGHT.pkl'), 'rb'), encoding='latin1')
        faces = model_data['f']
        self.focal_length=focal_length
        faces = np.concatenate([faces, faces_new], axis=0)
        
        self.camera_center = [self.image_size // 2, self.image_size // 2]
        faces_left = faces[:,[0,2,1]]
        
        self.register_buffer('faces', torch.from_numpy(faces))
        self.register_buffer('faces_left', torch.from_numpy(faces_left))
        
    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, is_weak_cam=False, ret_image=True, is_left=False, ):
        if not is_left:
            faces = self.faces
        else:
            faces = self.faces_left

        return super().forward(vertices, faces, landmarks, cameras, transform_matrix, focal_length, ret_image)