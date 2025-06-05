import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pointnet2_utils import PointNetSetAbstraction


class PointNet2Cls(nn.Module):
    def __init__(self,num_rc,num_class,model_size='large',radius=0.2,classify=False,split='early', lratio_known=False, boundary_known=False, lprotocol_known=False, normal_channel=True):
        super(PointNet2Cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.min_value = 0.4
        self.max_value = 1.0
        self.classify = classify
        self.split = split
        self.radius = radius
        one_hot_dim = 0
        if lratio_known:
            one_hot_dim = 5
        if boundary_known:
            one_hot_dim += 2
        if lprotocol_known:
            one_hot_dim += 3
        if model_size == 'large':
            #Normalized pointcloud (bnorm point to point distance is ~ 0.0004 - 0.0008)
            #radius = 0.2 (#0.002 iyi calismadi) (0.3 - 0.6 is the best, ondan sonra da 0.2 - 0.4)
            #radius = 0.4
            self.sa1 = PointNetSetAbstraction(npoint=128, radius=self.radius, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=64, radius=self.radius*2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
            
            self.feat_dim = 1024
            self.feat_dim2 = 64
            self.fc1 = nn.Linear(1024+one_hot_dim , 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(256, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.drop2 = nn.Dropout(0.4)
            self.fc3 = nn.Linear(64, num_rc)
            self.sigmoid_layer = nn.Sigmoid()
            
        elif model_size == 'small':
            #Normalized pointcloud
            self.sa1 = PointNetSetAbstraction(npoint=128, radius=self.radius, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
            self.sa2 = PointNetSetAbstraction(npoint=64, radius=self.radius*2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
            self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 256, 512], group_all=True)
            self.feat_dim = 512
            self.feat_dim2 = 64
            self.fc1 = nn.Linear(512+one_hot_dim , 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(256, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.drop2 = nn.Dropout(0.4)
            self.fc3 = nn.Linear(64, num_rc)
            self.sigmoid_layer = nn.Sigmoid()

        if self.classify is not None:
            if self.split == 'early':
                self.lprot_fc = nn.Sequential(
                    nn.Linear(self.feat_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_class)
                )
            elif self.split == 'late':
                self.lprot_fc = nn.Sequential(
                    nn.Linear(self.feat_dim2, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_class)
                )


    def forward(self, xyz, knn_idx, lratio=None, boundary=None, lprotocol=None):
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B,  self.feat_dim)
        if (self.classify is not None) and (self.split == 'early'):
            lprot_pred = self.lprot_fc(x)
        #concatenate lratio input (batch size x 5) and boundary input (batch size x 2) to x
        augmented_x = None
        if lratio is not None:
            augmented_x = torch.cat((x, lratio), dim=-1)
        if boundary is not None:
            augmented_x = torch.cat((augmented_x, boundary), dim=-1)
        if lprotocol is not None:
            augmented_x = torch.cat((augmented_x, lprotocol), dim=-1)
        if augmented_x is not None:
            x = self.drop1(F.relu(self.bn1(self.fc1(augmented_x))))
        else:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        if (self.classify is not None) and (self.split == 'late'):
            lprot_pred = self.lprot_fc(x)
        x = self.fc3(x)
        x = (self.sigmoid_layer(x) * (self.max_value - self.min_value)) + self.min_value 
        if self.classify is not None:
            return x, lprot_pred
        return x  #l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss