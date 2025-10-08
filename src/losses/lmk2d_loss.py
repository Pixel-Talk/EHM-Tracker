import torch
import torch.nn as nn
import numpy as np

class Landmark2DLoss(nn.Module):
    def __init__(self, left_indices=None, right_indices=None, front_indices=None, 
                 selected_mp_indices=None, metric='l1', **kwargs) -> None:
        super().__init__()
        if metric == 'robust':
            from .robust_loss import GMoF
            self.metric = GMoF(rho=kwargs.get('rho', 1.0))
        elif metric == 'l2':
            self.metric = nn.MSELoss()
        else:
            self.metric = nn.L1Loss()

        self.left_indices  = torch.tensor(left_indices) if left_indices is not None else None
        self.right_indices = torch.tensor(right_indices) if right_indices is not None else None
        self.front_indices = torch.tensor(front_indices) if front_indices is not None else None
        self.selected_mp_indices = torch.tensor(selected_mp_indices) if selected_mp_indices is not None else None
    
    def forward(self, x:torch.Tensor, y:torch.Tensor, cam=None, weight=None):
        """calc face landmark loss

        Args:
            x (torch.Tensor): [B, N, x]
            y (torch.Tensor): [B, N, x]
            cam (torch.Tensor, optional): [B, 3]. Defaults to None.

        Returns:
            torch.Tensor: loss value
        """

        _x, _y = x, y

        if _x.shape[1] == 203:
            # return 0
            assert cam is not None
            assert self.left_indices is not None and self.right_indices is not None and self.front_indices is not None
            t_loss = 0
            t_x, t_y = _x[cam[:, 1] < -0.05], _y[cam[:, 1] < -0.05]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.left_indices], t_y[:, self.left_indices])
            t_x, t_y = x[cam[:, 1] > 0.05], y[cam[:, 1] > 0.05]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.right_indices], t_y[:, self.right_indices])
            mask = (cam[:, 1] >= -0.05) & (cam[:, 1] <= 0.05)
            t_x, t_y = x[mask], y[mask]
            if t_x.shape[0] > 0:
                t_loss += self.metric(t_x[:, self.front_indices], t_y[:, self.front_indices])
            return (t_loss / _x.shape[0]) * 15  # trust mouth region
        elif _x.shape[1] == 478:
            return self.metric(_x[:, self.selected_mp_indices], _y) / 5
        elif _x.shape[1] == 105 and _y.shape[1] == 478:
            return self.metric(_x, _y[:, self.selected_mp_indices]) / 5
        elif _x.shape[1] != _y.shape[1]:
            min_len = min(_x.shape[1], _y.shape[1])
            return self.metric(_x[:, :min_len], _y[:, :min_len]) / 5
        else:
            if weight is None:
                return self.metric(_x, _y).float()
            else:
                return self.metric(_x * weight, _y * weight).float()
