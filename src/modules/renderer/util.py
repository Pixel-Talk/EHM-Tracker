import numpy as np
import torch
import torch.nn.functional as F

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    
def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


from pytorch3d.renderer import look_at_view_transform 

def weak_cam2persp_cam(wcam, focal_length=12, z_dist=10):
    """_summary_

    Args:
        wcam (torch.Tensor): In shape Bx3, for each [s, x, y]
        focal_length (int, optional): perspective camera focal length. Defaults to 12.
        z_dist (int, optional): perspective camera at (0, 0, z). Defaults to 10.

    Returns:
        R, T: Rotation matrix and translation vector
    """
    bz = wcam.shape[0]
    R, T = look_at_view_transform(dist=z_dist, device=wcam.device)
    R = R.repeat(bz, 1, 1)
    T = T.repeat(bz, 1)
    T[:, 2] = focal_length / wcam[:, 0]
    T[:, 1] = wcam[:, 2]
    T[:, 0] = -wcam[:, 1]
    return R, T

def cam2persp_cam_fov(wcam, tanfov=1/12):
    """
    scale by changing zdist,unchange fov
    Returns: w2c
        R, T: Rotation matrix and translation vector
    """
    #blender coord to pytorch3d coord
    flx=-1.0
    fly=1.0
    flz=-1.0
    bz = wcam.shape[0]
    z_dist=1/tanfov
    R=torch.tensor([[flx,0,0],
                    [0,fly,0],
                    [0,0,flz]],device=wcam.device,dtype=torch.float32)
    T=torch.tensor([0,0,z_dist],device=wcam.device,dtype=torch.float32)
    R = R.repeat(bz, 1, 1)
    T = T.repeat(bz, 1)
    T[:, 2] = T[:, 2] / wcam[:, 0]
    T[:, 1] = wcam[:, 2] *fly
    T[:, 0] = wcam[:, 1] *flx
    return R, T

def cam2persp_cam_fov_body(wcam, tanfov=1/12):
    """
    scale by changing zdist,unchange fov
    Returns: w2c
        R, T: Rotation matrix and translation vector
    """
    #image coord to pytorch3d coord
    flx=-1.0
    fly=-1.0
    flz=1.0
    bz = wcam.shape[0]
    z_dist=1/tanfov
    R=torch.tensor([[flx,0,0],
                    [0,fly,0],
                    [0,0,flz]],device=wcam.device,dtype=torch.float32)
    T=torch.tensor([0,0,z_dist],device=wcam.device,dtype=torch.float32)
    R = R.repeat(bz, 1, 1)
    T = T.repeat(bz, 1)
    T[:, 2] = T[:, 2] / wcam[:, 0]
    T[:, 1] = wcam[:, 2]*fly 
    T[:, 0] = wcam[:, 1]*flx
    return R, T