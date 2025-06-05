import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pointnet2_utils import index_points, square_distance


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, init_dim=3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(init_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    #original  
    #xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features, knn_idx=None):
        if knn_idx is None:
            dists = square_distance(xyz, xyz)
            knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        else:
            knn_idx = knn_idx[:,:, :self.k]
        knn_xyz = index_points(xyz, knn_idx)
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        #pos_enc = self.fc_delta(xyz[:, :, None]) 
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

    # def forward(self, xyz, features, knn_idx=None):
    #     if knn_idx is None:
    #         dists = square_distance(xyz, xyz)
    #         knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
    #     else:
    #         knn_idx = knn_idx[:,:, :self.k]
    #     knn_xyz = index_points(xyz, knn_idx)
    #     pre = features
    #     x = self.fc1(features)
    #     q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
    #     pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
    #     attn = torch.matmul(q[:, :, None].repeat(1,1,k.shape[-2], 1), k.transpose(-2, -1))[:,:,0,:]
    #     attn = self.fc_gamma(attn[:, :, :, None].repeat(1,1,1,k.shape[-1]) + pos_enc)
    #     attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
    #     res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
    #     res = self.fc2(res) + pre
    #     return res, attn
    