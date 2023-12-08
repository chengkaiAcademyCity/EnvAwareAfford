"""
    This file borrows PointNet2 implementation: https://github.com/erikwijmans/Pointnet2_PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import numpy as np

class MyFPModule(nn.Module):
    def __init__(self):
        super(MyFPModule, self).__init__()

    # B x N x 3, B x M X 3, B x F x M
    # output: B x F x N
    def forward(self, unknown, known, known_feats):
        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(
            known_feats, idx, weight
        )

        new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        return new_features.squeeze(-1)


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class PointNet2SemSegSSGShape(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 256],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]), self.fc_layer2(bottleneck_feats)


# class PointNet(nn.Module):
#     def __init__(self, feat_dim):
#         super(PointNet, self).__init__()
#
#         self.conv1 = nn.Conv1d(feat_dim*2, feat_dim, 1)
#         self.conv2 = nn.Conv1d(feat_dim, feat_dim, 1)
#         self.conv3 = nn.Conv1d(feat_dim, feat_dim, 1)
#
#         self.bn1 = nn.BatchNorm1d(feat_dim)
#         self.bn2 = nn.BatchNorm1d(feat_dim)
#         self.bn3 = nn.BatchNorm1d(feat_dim)
#
#     # B x 2F x N
#     # output: B x F
#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = x.max(dim=-1)[0]
#         return x

class PointNet(nn.Module):
    def __init__(self, feat_dim, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, feat_dim)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x #, trans_feat #B x 1024, B x 64 x 64
class Critic(nn.Module):
    def __init__(self, feat_dim):
        super(Critic, self).__init__()

        self.mlp1 = nn.Linear(feat_dim+feat_dim+3, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, 1)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')

    # B x 3F
    # output: B
    def forward(self, net):
        net = F.leaky_relu(self.mlp1(net))
        net = self.mlp2(net).squeeze(-1)
        return net
     
    # cross entropy loss
    def get_ce_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss(pred_logits, gt_labels.float())
        return loss


class Network(nn.Module):
    def __init__(self, feat_dim):
        super(Network, self).__init__()
        
        self.object_pointnet2 = PointNet2SemSegSSGShape({'feat_dim': feat_dim})
        self.pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})
        
        self.pointnet = PointNet(feat_dim)
        self.critic = Critic(feat_dim)
        
        self.fp_layer = MyFPModule()


    # scene_pcs: B x N x 3 (float), with the 0th point to be the query point
    # acting_pcs: B x M x 3 (float)
    # pred_result_logits: B, whole_feats: B x F x N, acting_feats: B x F
    def forward(self, scene_pcs, robot_p, gt_object_pc = None):

        whole_feats = self.pointnet2(scene_pcs.repeat(1, 1, 2))     # B x F x N

        gripper_p = robot_p
        gripper_p[:, 2] += 1

        DisField_vec1 = scene_pcs - scene_pcs[:, 0:1, :]
        DisField_vec2 = scene_pcs - gripper_p.unsqueeze(1).repeat(1, scene_pcs.shape[1], 1)
        DisField = torch.cross(DisField_vec1, DisField_vec2, dim=2)  # B x N x 3

        FieldDis = torch.norm(DisField, dim=2)
        FieldDis[gt_object_pc] = 10000

        t50dist, t50idx = torch.topk(FieldDis, 50, dim=1, largest=False) # B x 50

        DisField_pcs = torch.gather(DisField, 1, t50idx.unsqueeze(-1).repeat(1, 1, 3))  # B x 50 x 3

        torch.cuda.empty_cache()
        # use a simple pointnet with max-pooling to get feat
        DisF_global_feats = self.pointnet(DisField_pcs.permute(0, 2, 1))  # B x F
        torch.cuda.empty_cache()

        # gather all useful feats
        all_feats = torch.cat([whole_feats[:, :, 0], DisF_global_feats, robot_p], dim=1)   # B x (3+2F)
        
        # use MLP to predict final logits/scores
        pred_result_logits = self.critic(all_feats)     # B

        end_points = dict()
        torch.cuda.empty_cache()
        return pred_result_logits, DisF_global_feats, end_points

    # scene_pcs: B x N x 3 (float)
    # acting_pcs: B x M x 3 (float)
    # To be updated
    def inference_whole_pc(self, scene_pcs, robot_p, gt_object_pc = None):
        with torch.no_grad():
            torch.cuda.empty_cache()
            batch_size = scene_pcs.shape[0]
            num_scene_pts = scene_pcs.shape[1]
            torch.cuda.empty_cache()
            whole_feats = self.pointnet2(scene_pcs.repeat(1, 1, 2))     # B x F x N
            torch.cuda.empty_cache()

            gripper_p = robot_p
            gripper_p[:, 2] += 1

            DisField_vec1 = scene_pcs.unsqueeze(1).repeat(1, num_scene_pts, 1, 1) - scene_pcs.unsqueeze(2).repeat(1, 1, num_scene_pts, 1) # B x N x N x 3
            DisField_vec2 = scene_pcs.unsqueeze(1).repeat(1, num_scene_pts, 1, 1) - gripper_p.unsqueeze(1).unsqueeze(1).repeat(1, num_scene_pts, num_scene_pts, 1) # B x N x N x 3
            DisField = torch.cross(DisField_vec1, DisField_vec2, dim=3)  # B x N x N x 3
            torch.cuda.empty_cache()

            FieldDis = torch.norm(DisField, dim=3)
            gt_object_pc = gt_object_pc.unsqueeze(1).repeat(1, num_scene_pts, 1)
            FieldDis[gt_object_pc] = 10000

            t50dist, t50idx = torch.topk(FieldDis, 50, dim=2, largest=False) # B x N x 50
            torch.cuda.empty_cache()

            DisField_pcs = torch.gather(DisField, 2, t50idx.unsqueeze(-1).repeat(1, 1, 1, 3))  # B x N x 50 x 3
            torch.cuda.empty_cache()

            DisField_pcs = DisField_pcs.view(batch_size * num_scene_pts, 50, 3) # BN x 50 x 3
            # print(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
            print(DisField_pcs)
        # print('clean:',torch.cuda.memory_summary())
            DisF_global_feats = self.pointnet(DisField_pcs.permute(0, 2, 1))  # BN x F
            torch.cuda.empty_cache()
            DisF_global_feats = DisF_global_feats.view(batch_size, num_scene_pts, -1) # B x N x F

            all_feats = torch.cat([whole_feats.permute(0,2,1), DisF_global_feats, robot_p.repeat(1, num_scene_pts, 1)], dim=2)   # B x N x (3+2F)

            torch.cuda.empty_cache()
            pred_result_logits = self.critic(all_feats)     # B x N
            torch.cuda.empty_cache()

            whole_pred_results = torch.sigmoid(pred_result_logits)
            torch.cuda.empty_cache()
        return whole_pred_results

