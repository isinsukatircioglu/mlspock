import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction
from model.transformer import TransformerBlock


class PointTransformerCls(nn.Module):
    def __init__(self,num_class,d_points, transformer_dim, num_neigh, model_size='large', radius=0.2, classify=False, split='early', lratio_known=False, boundary_known=False, lprotocol_known=False, dimension_known=False, normal_channel=True):
        super(PointTransformerCls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.min_value = 0.4
        self.max_value = 1.0
        self.tr1_point_dim = 32
        self.num_neigh = num_neigh
        self.classify = classify
        self.split = split
        self.radius = radius
        self.fc_transformer = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        one_hot_dim = 0
        if lratio_known:
            one_hot_dim = 5
        if boundary_known:
            one_hot_dim += 2
        if lprotocol_known:
            one_hot_dim += 3
        if dimension_known:
            one_hot_dim += 4
        #Keep this
        if model_size == 'large':
            #Normalized pointcloud
            self.sa1 = PointNetSetAbstraction(npoint=128, radius=self.radius, nsample=32, in_channel=in_channel+self.tr1_point_dim, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=64, radius=self.radius*2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

            self.transformer1 = TransformerBlock(self.tr1_point_dim, transformer_dim, num_neigh)
            self.transformer2 = TransformerBlock(128, transformer_dim, num_neigh)
            self.transformer3 = TransformerBlock(256, transformer_dim, num_neigh)
            self.transformer4 = TransformerBlock(1024, transformer_dim, num_neigh)

            self.feat_dim = 1024
            self.fc2 = nn.Sequential(
            nn.Linear(1024+one_hot_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
            )
            self.sigmoid_layer = nn.Sigmoid()
        elif model_size == 'small':
            self.sa1 = PointNetSetAbstraction(npoint=128, radius=self.radius, nsample=32, in_channel=in_channel+self.tr1_point_dim, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=64, radius=self.radius*2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
            
            self.feat_dim = 512
            self.transformer1 = TransformerBlock(self.tr1_point_dim, transformer_dim, num_neigh)
            self.transformer2 = TransformerBlock(128, transformer_dim, num_neigh)
            self.transformer3 = TransformerBlock(256, transformer_dim, num_neigh)
            self.transformer4 = TransformerBlock(512, transformer_dim, num_neigh)
            self.fc2 = nn.Sequential(
            nn.Linear(512+one_hot_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_class)
            )
            self.sigmoid_layer = nn.Sigmoid()
        
        # if self.classify is not None:
        #     if self.split == 'early':
        #         self.lprot_fc = nn.Sequential(
        #             nn.Linear(self.feat_dim, 64),
        #             nn.ReLU(),
        #             nn.Linear(64, num_class)
        #         )
        #     elif self.split == 'late':
        #         self.lprot_fc = nn.Sequential(
        #             nn.Linear(self.feat_dim2, 32),
        #             nn.ReLU(),
        #             nn.Linear(32, num_class)
        #         )

    def forward(self, xyz, knn_idx, lratio=None, boundary=None, lprotocol=None, column_size=None):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        tr1_points = self.transformer1(xyz, self.fc_transformer(xyz), knn_idx)[0]
        l1_xyz, l1_points = self.sa1(xyz, tr1_points)
        tr2_points = self.transformer2(l1_xyz, l1_points)[0]
        l2_xyz, l2_points = self.sa2(l1_xyz, tr2_points)
        tr3_points = self.transformer3(l2_xyz, l2_points)[0]
        l3_xyz, l3_points = self.sa3(l2_xyz, tr3_points)
        tr4_points = self.transformer4(l3_xyz, l3_points)[0]
        x = tr4_points.view(B, self.feat_dim)
        # if (self.classify is not None) and (self.split == 'early'):
        #     lprot_pred = self.lprot_fc(x)
        #concatenate lratio input (batch size x 5) and boundary input (batch size x 2) to x
        augmented_x = None
        if lratio is not None:
            augmented_x = torch.cat((x, lratio), dim=-1)
        if boundary is not None:
            if augmented_x is None:
                augmented_x = torch.cat((x, boundary), dim=-1)
            else:
                augmented_x = torch.cat((augmented_x, boundary), dim=-1)
        if lprotocol is not None:
            if augmented_x is None:
                augmented_x = torch.cat((x, lprotocol), dim=-1)
            else:
                augmented_x = torch.cat((augmented_x, lprotocol), dim=-1)
        if column_size is not None:
            if augmented_x is None:
                augmented_x = torch.cat((x, column_size), dim=-1)
            else:
                augmented_x = torch.cat((augmented_x, column_size), dim=-1)
        if augmented_x is not None:
            x = self.fc2(augmented_x)
        else:
            x = self.fc2(x)
        # if (self.classify is not None) and (self.split == 'late'):
        #     lprot_pred = self.lprot_fc(x)
        x = (self.sigmoid_layer(x) * (self.max_value - self.min_value)) + self.min_value 
        #x = F.log_softmax(x, -1)
        # if self.classify is not None:
        #     return x, lprot_pred
        return x #, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss