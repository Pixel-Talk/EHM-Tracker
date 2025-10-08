import torch
import pickle
import os.path as osp
from pytorch3d.io import load_obj
from ...utils.helper import face_vertices
from ...utils.graphics import GS_BaseMeshRenderer

class Renderer(GS_BaseMeshRenderer):
    def __init__(self, assets_dir, image_size=512, device='cuda', focal_length=24):
        super().__init__( image_size, focal_length=focal_length)
        obj_filename = osp.join(assets_dir, 'head_template.obj')
        self.focal_length=focal_length
        verts, faces, aux = load_obj(obj_filename)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
        faces = faces.verts_idx[None,...]

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.

        flame_masks = pickle.load(
            open(osp.join(assets_dir, 'FLAME_masks/FLAME_masks.pkl'), 'rb'),
            encoding='latin1')
        self.flame_masks = flame_masks

        self.register_buffer('faces', faces)

        face_colors = face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)
        
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1) #[bz, ntv, 3]
        uvcoords = uvcoords*2 - 1; uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        
    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, is_weak_cam=False, ret_image=True):
        if faces is None:
            faces = self.faces.squeeze(0)
        return super().forward(vertices, faces, landmarks, cameras, transform_matrix, focal_length, ret_image)