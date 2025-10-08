from .lmk2d_loss import Landmark2DLoss
from .pose_loss import PoseLoss
from torch.nn import functional as F

def l2_distance(x, y):
    return F.mse_loss(x, y)


def l1_distance(x, y):
    return F.l1_loss(x, y)



