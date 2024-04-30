import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from model.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class PointNetCls(nn.Module):
    def __init__(self, k=2, normal_channel=False):
        super(PointNetCls, self).__init__()
       
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.min_value = 0.4
        self.max_value = 1.0
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel) #global_feat=True, feature_transform=True
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k)
        # self.dropout = nn.Dropout(p=0.4)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.relu = nn.ReLU()
        # self.sigmoid_layer = nn.Sigmoid()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim=1)
        x = (self.sigmoid_layer(x) * (self.max_value - self.min_value)) + self.min_value 
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss