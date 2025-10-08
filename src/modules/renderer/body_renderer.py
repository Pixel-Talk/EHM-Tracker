import os.path as osp
import torch
from pytorch3d.io import load_obj
from ...utils.graphics import GS_BaseMeshRenderer
from ...utils.helper import face_vertices

class Renderer(GS_BaseMeshRenderer):
    def __init__(self, assets_dir, image_size=1024, device='cuda', focal_length=24):
        super().__init__(
            image_size, focal_length=focal_length, inverse_light=True
        )
        topology_path = osp.join(assets_dir, 'smplx_tex.obj')
        self.focal_length = focal_length
        verts, faces, aux = load_obj(topology_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1] * 0.0 + 1.0], -1
        )  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        
    def forward(
        self,
        vertices,
        faces=None,
        landmarks={},
        cameras=None,
        transform_matrix=None,
        focal_length=None,
        is_weak_cam=False,
        ret_image=True
    ):
        if faces is None:
            faces = self.faces.squeeze(0)
        return super().forward(
            vertices, faces, landmarks, cameras, transform_matrix,
            focal_length, ret_image
        )