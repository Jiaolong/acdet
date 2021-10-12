import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as f


class MetaKernel(nn.Module):
    def __init__(self, kernel_cfg):
        super().__init__()
        self.meta_channels = kernel_cfg.META_CHANNELS
        self.in_channels = kernel_cfg.INPUT_CHANNELS
        self.out_channels = kernel_cfg.OUTPUT_CHANNELS
        self.feature_map_size = kernel_cfg.FEATURE_MAP_SIZE

        self.weight_mlp1 = nn.Linear(self.meta_channels, self.in_channels, bias=False)
        self.weight_bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.use_mask = kernel_cfg.USE_MASK
        self.use_attention = kernel_cfg.USE_ATTENTION
        self.reduced = kernel_cfg.REDUCED
        self.residual=kernel_cfg.get('RESIDUAL',False)
        self.remask=kernel_cfg.get('REMASK',False)

        if self.reduced:
            assert self.use_mask, "reduced must required use_mask is True"
        if self.use_attention:
            self.weight_mlp2 = nn.Linear(self.in_channels, 1)
        else:
            self.weight_mlp2 = nn.Linear(self.in_channels, self.in_channels)

        self.aggregation_mlp = nn.Linear(
            9 * self.in_channels, self.out_channels, bias=False)
        self.aggregation_bn = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.fold = nn.Fold(self.feature_map_size, kernel_size=1,
                            dilation=1, padding=0)

    def forward(self, x, mask=None):
        if self.reduced:
            return self.forward_v2(x, mask)
        else:
            return self.forward_v1(x, mask)

    def forward_v1(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H*W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C
        features_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        x_pn = x_unfold[..., 0:3]  # B*HW*9*3

        if mask is not None and self.use_mask:
            # B*1*H*W ---> B*(1*3*3)*(H*W)
            m_unfold = self.unfold(mask.float())
            m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
                (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
            m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        x_pn_range = torch.norm(x_pn, p=2, dim=-1).unsqueeze(-1)
        x_pn = torch.cat((x_pn, x_pn_range), dim=-1)
        x_p0 = x_pn[:, :, 4:5, :]  # B*HW*1*4
        pn_p0 = x_pn-x_p0  # B*HW*9*4
        weights = self.weight_mlp1(pn_p0)  # B*HW*9*C'
        weights = weights.reshape(
            (batch_size, -1, self.in_channels)).transpose(1, 2).contiguous()  # B*C'*HW9
        weights = self.weight_bn1(weights)
        weights = weights.transpose(1, 2).contiguous().reshape(
            (batch_size, -1, 9, self.in_channels))  # B*HW*9*C'
        weights = self.relu1(weights)
        weights = self.weight_mlp2(weights)

        # set weights=0 for empty voxels
        if mask is not None and self.use_mask:
            weights = weights * m_unfold

        if self.use_attention:
            weights = weights.squeeze(-1)
            weights = f.softmax(weights, dim=-1)
            weights = weights.unsqueeze(-1)

        if self.remask:
            weights=weights*m_unfold

        if self.residual:
            weights=1+weights

        features_unfold = weights*features_unfold  # B*HW*9*C'

        features_unfold = features_unfold.reshape(
            (batch_size, -1, 9*self.in_channels))  # B*HW*9C'

        features_unfold = self.aggregation_mlp(features_unfold)  # B*HW*C''
        features_unfold = features_unfold.transpose(
            1, 2).contiguous()  # B*C''*HW
        features_unfold = self.aggregation_bn(features_unfold)
        features_unfold = self.relu2(features_unfold)
        features = self.fold(features_unfold)

        return features

    def forward_v2(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H*W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C
        features_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        x_pn = x_unfold[..., 0:3]  # B*HW*9*3

        if mask is not None and self.use_mask:
            # B*1*H*W ---> B*(1*3*3)*(H*W)
            m_unfold = self.unfold(mask.float())
            m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
                (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
            m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        x_pn_range = torch.norm(x_pn, p=2, dim=-1).unsqueeze(-1)
        x_pn = torch.cat((x_pn, x_pn_range), dim=-1)
        x_p0 = x_pn[:, :, 4:5, :]  # B*HW*1*4
        pn_p0 = x_pn-x_p0  # B*HW*9*4
        pn_p0 = pn_p0.reshape(-1, pn_p0.shape[-1]).contiguous()
        
        weights_reduce = self.weight_mlp1(pn_p0[m_unfold.view(-1) > 0])  # B*HW*9*C'
        weights_reduce = self.weight_bn1(weights_reduce)
        weights_reduce = self.relu1(weights_reduce)
        weights_reduce = self.weight_mlp2(weights_reduce)

        weights = m_unfold.new_zeros(batch_size, H * W, 9, weights_reduce.shape[-1])
        weights[m_unfold[..., 0] > 0] = weights_reduce

        if self.use_attention:
            weights = weights.squeeze(-1)
            weights = f.softmax(weights, dim=-1)
            weights = weights.unsqueeze(-1)
        if self.remask:
            weights=weights*m_unfold
        if self.residual:
            weights=1+weights
        features_unfold = weights*features_unfold  # B*HW*9*C'

        features_unfold = features_unfold.reshape(
            (batch_size, -1, 9*self.in_channels))  # B*HW*9C'

        features_unfold = self.aggregation_mlp(features_unfold)  # B*HW*C''
        features_unfold = features_unfold.transpose(
            1, 2).contiguous()  # B*C''*HW
        features_unfold = self.aggregation_bn(features_unfold)
        features_unfold = self.relu2(features_unfold)
        features = self.fold(features_unfold)

        return features


class EdgeConvKernel(nn.Module):
    def __init__(self, kernel_cfg):
        super().__init__()

        self.in_channels = kernel_cfg.INPUT_CHANNELS
        self.out_channels = kernel_cfg.OUTPUT_CHANNELS
        self.feature_map_size = kernel_cfg.FEATURE_MAP_SIZE
        self.use_mask = kernel_cfg.USE_MASK
        self.reduced = kernel_cfg.REDUCED
        if self.reduced:
            assert self.use_mask, "reduced must required use_mask is True"
        self.mlp1 = nn.Linear(3 + 2 * self.in_channels, self.in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.mlp2 = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=9, stride=9)

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.fold = nn.Fold(self.feature_map_size, kernel_size=1,
                            dilation=1, padding=0)

    def forward(self, x, mask=None):
        if self.reduced:
            return self.forward_v2(x, mask)
        else:
            return self.forward_v1(x, mask)

    def forward_v1(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H * W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C

        feature_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        feature_x0_unfold = feature_unfold[:, :, 4:5].expand(-1, -1, 9, -1)

        if mask is not None and self.use_mask:
            # B*1*H*W ---> B*(1*3*3)*(H*W)
            m_unfold = self.unfold(mask.float())
            m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
                (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
            m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        x_pn = x_unfold[..., 0:3]  # B*HW*9*3
        x_pn_azimuth = torch.atan2(
            x_pn[..., 1], x_pn[..., 0]).unsqueeze(-1)  # B*HW*9*1
        x_p0_azimuth = x_pn_azimuth[:, :, 4:5]  # B*HW*1*1
        x_pn_range = torch.norm(
            x_pn[..., :3], p=2, dim=-1).unsqueeze(-1)  # B*HW*9*1
        x_p0_range = x_pn_range[:, :, 4:5]  # B*HW*1*1
        x_pn_depth = torch.norm(x_pn[..., :2], p=2, dim=-1)  # B*HW*9*1
        x_pn_inclination = torch.atan2(
            x_pn[..., 2], x_pn_depth).unsqueeze(-1)  # B*HW*9*1
        x_p0_inclination = x_pn_inclination[:, :, 4:5]  # B*HW*1*1
        x_pn_delta_azi = x_pn_azimuth-x_p0_azimuth  # B*HW*9*1
        x_pn_delta_inc = x_pn_inclination-x_p0_inclination  # B*HW*9*1

        gamma0 = x_pn_range * \
            torch.cos(x_pn_delta_azi)*torch.cos(x_pn_delta_inc) - \
            x_p0_range  # B*HW*9*1
        gamma1 = x_pn_range * \
            torch.cos(x_pn_delta_azi)*torch.sin(x_pn_delta_inc)  # B*HW*9*1
        gamma2 = x_pn_range*torch.sin(x_pn_delta_azi)  # B*HW*9*1
        gamma = torch.cat([gamma0, gamma1, gamma2], dim=-1)  # B*HW*9*3

        pn_p0 = torch.cat([feature_unfold, feature_x0_unfold, gamma], dim=-1)

        feature_unfold = self.mlp1(pn_p0)  # B*HW*9*C'
        feature_unfold = feature_unfold.reshape(
            (batch_size, -1, self.in_channels)).transpose(1, 2).contiguous()  # B*C'*HW9
        feature_unfold = self.bn1(feature_unfold)  # B*C'*HW9
        feature_unfold = self.relu1(feature_unfold)  # B*C'*HW9
        feature_unfold = feature_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H*W, 9, self.in_channels))

        if mask is not None and self.use_mask:
            feature_unfold = feature_unfold*m_unfold

        feature_unfold = feature_unfold.permute(
            0, 3, 1, 2).contiguous().reshape((batch_size, -1, 9))  # B*C'HW*9
        feature_unfold = self.maxpool(feature_unfold).squeeze(-1)  # B*C'HW

        feature_unfold = feature_unfold.reshape(
            (batch_size, self.in_channels, H*W)).transpose(1, 2).contiguous()  # B*HW*C'
        feature_unfold = self.mlp2(feature_unfold)
        feature_unfold = feature_unfold.transpose(1, 2)
        feature_unfold = self.bn2(feature_unfold)
        feature_unfold = self.relu2(feature_unfold)

        features = self.fold(feature_unfold)  # B*C''*H*W

        return features

    def forward_v2(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H * W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C

        features_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        features_x0_unfold = features_unfold[:, :,
                                             4:5].expand(-1, -1, 9, -1)  # B*HW*1*C'
        x_pn = x_unfold[..., 0:3]  # B*HW*9*3

        m_unfold = self.unfold(mask.float())  # B*1*H*W ---> B*(1*3*3)*(H*W)
        m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
        m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        # mask_unfold = m_unfold[:, :, 4, 0].bool()  # B*HW
        # m_unfold = m_unfold[mask_unfold]  # N*9*1
        # features_unfold = features_unfold[mask_unfold]  # N*9*C'
        # features_x0_unfold = features_x0_unfold[mask_unfold]  # N*9*C'
        # x_pn = x_pn[mask_unfold]  # N*9*3

        x_pn_azimuth = torch.atan2(
            x_pn[..., 1], x_pn[..., 0]).unsqueeze(-1)  # B*HW*9*1
        x_p0_azimuth = x_pn_azimuth[:,:, 4:5]  #  B*HW*1*1
        x_pn_range = torch.norm(
            x_pn[..., :3], p=2, dim=-1).unsqueeze(-1)  # B*HW*9*1
        x_p0_range = x_pn_range[:,:, 4:5]  # B*HW*1*1
        x_pn_depth = torch.norm(x_pn[..., :2], p=2, dim=-1)  # B*HW*9
        x_pn_inclination = torch.atan2(
            x_pn[..., 2], x_pn_depth).unsqueeze(-1)  # B*HW*9*1
        x_p0_inclination = x_pn_inclination[:, 4:5]  # B*HW*1*1
        x_pn_delta_azi = x_pn_azimuth-x_p0_azimuth  # B*HW*9*1
        x_pn_delta_inc = x_pn_inclination-x_p0_inclination  # B*HW*9*1

        gamma0 = x_pn_range * \
            torch.cos(x_pn_delta_azi)*torch.cos(x_pn_delta_inc) - \
            x_p0_range  # B*HW*9*1
        gamma1 = x_pn_range * \
            torch.cos(x_pn_delta_azi)*torch.sin(x_pn_delta_inc)  # B*HW*9*1
        gamma2 = x_pn_range*torch.sin(x_pn_delta_azi)  # B*HW*9*1
        gamma = torch.cat([gamma0, gamma1, gamma2], dim=-1)  # B*HW*9*1

        pn_p0 = torch.cat([features_unfold, features_x0_unfold, gamma], dim=-1)

        pn_p0_reduce=pn_p0[m_unfold[...,0]>0].contiguous() #N*C

        features_unfold_reduce = self.mlp1(pn_p0_reduce)  # N*C'
        # features_unfold_reduce = features_unfold_reduce.transpose(
        #     1, 2).contiguous()  # N*C'*9
        features_unfold_reduce = self.bn1(features_unfold_reduce)  # N*C'
        features_unfold_reduce = self.relu1(features_unfold_reduce)  # N*C'
        # features_unfold_reduce = features_unfold_reduce.transpose(
        #     1, 2).contiguous()  # N*9*C'

        features_unfold = features_unfold_reduce.new_zeros(batch_size, H * W, 9, features_unfold_reduce.shape[-1])
        features_unfold[m_unfold[...,0] > 0] = features_unfold_reduce  #B*HW*9*C
        # if mask is not None and self.use_mask:
        # features_unfold = features_unfold*m_unfold

        features_unfold = features_unfold.transpose(
            2, 3).contiguous().reshape(batch_size,-1,9)  #B*HWC*9
        features_unfold = self.maxpool(features_unfold).squeeze(-1)  # B*HWC

        features_unfold=features_unfold.reshape(batch_size,H*W,-1) # B*HW*C'
        # print("feature shape is ",feature_unfold.shape)
        features_unfold = self.mlp2(features_unfold)
        features_unfold=features_unfold.transpose(1,2).contiguous()
        features_unfold = self.bn2(features_unfold)
        features_unfold = self.relu2(features_unfold)


        # features = features_unfold.transpose(1, 2).contiguous()
        features = self.fold(features_unfold)  # B*C''*H*W

        return features


class MetaKernelDualAtt(nn.Module):
    def __init__(self, kernel_cfg):
        super().__init__()
        self.meta_channels = kernel_cfg.META_CHANNELS
        self.in_channels = kernel_cfg.INPUT_CHANNELS
        self.out_channels = kernel_cfg.OUTPUT_CHANNELS
        self.feature_map_size = kernel_cfg.FEATURE_MAP_SIZE

        self.weight_mlp_channel= nn.Linear(self.meta_channels, self.in_channels)
        self.weight_mlp_spatial = nn.Linear(self.meta_channels, self.in_channels)


        self.use_mask = kernel_cfg.USE_MASK
        self.reduced = kernel_cfg.REDUCED
        self.residual = kernel_cfg.get('RESIDUAL', False)

        if self.reduced:
            assert self.use_mask, "reduced must required use_mask is True"

        self.aggregation_mlp = nn.Linear(
            9 * self.in_channels, self.out_channels, bias=False)
        self.aggregation_bn = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)
        self.fold = nn.Fold(self.feature_map_size, kernel_size=1,
                            dilation=1, padding=0)

    def forward(self, x, mask=None):
        if self.reduced:
            return self.forward_v2(x, mask)
        else:
            return self.forward_v1(x, mask)

    def forward_v1(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H * W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C
        features_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        x_pn = x_unfold[..., 0:3]  # B*HW*9*3

        if mask is not None and self.use_mask:
            # B*1*H*W ---> B*(1*3*3)*(H*W)
            m_unfold = self.unfold(mask.float())
            m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
                (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
            m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        x_pn_range = torch.norm(x_pn, p=2, dim=-1).unsqueeze(-1)
        x_pn = torch.cat((x_pn, x_pn_range), dim=-1)
        x_p0 = x_pn[:, :, 4:5, :]  # B*HW*1*4
        pn_p0 = x_pn - x_p0  # B*HW*9*4
        weights_spatial = self.weight_mlp_spatial(pn_p0)  # B*HW*9*C'
        weights_channel = self.weight_mlp_channel(pn_p0)  # B*HW*9*C'
        weights_spatial,_=torch.max(weights_spatial,dim=3,keepdim=True) # B*HW*9*1'
        weights_channel,_=torch.max(weights_channel,dim=2,keepdim=True) # B*HW*1*C'
        weights_spatial=f.softmax(weights_spatial,dim=2)
        weights_channel=f.softmax(weights_channel,dim=3)

        weights=weights_spatial+weights_channel

        # set weights=0 for empty voxels
        if mask is not None and self.use_mask:
            weights = weights * m_unfold

        if self.residual:
            weights=weights+1

        features_unfold = weights * features_unfold  # B*HW*9*C'

        features_unfold = features_unfold.reshape(
            (batch_size, -1, 9 * self.in_channels))  # B*HW*9C'

        features_unfold = self.aggregation_mlp(features_unfold)  # B*HW*C''
        features_unfold = features_unfold.transpose(
            1, 2).contiguous()  # B*C''*HW
        features_unfold = self.aggregation_bn(features_unfold)
        features_unfold = self.relu2(features_unfold)
        features = self.fold(features_unfold)

        return features

    def forward_v2(self, x, mask=None):

        batch_size, _, H, W = x.shape

        x_unfold = self.unfold(x)  # B*C*H*W ---> B*(C*3*3)*(H*W)
        x_unfold = x_unfold.transpose(1, 2).contiguous().reshape(
            (batch_size, H * W, x.size(1), -1))  # B*HW*C*9
        x_unfold = x_unfold.transpose(2, 3).contiguous()  # B*HW*9*C
        features_unfold = x_unfold[..., 3:]  # B*HW*9*C'
        x_pn = x_unfold[..., 0:3]  # B*HW*9*3

        if mask is not None and self.use_mask:
            # B*1*H*W ---> B*(1*3*3)*(H*W)
            m_unfold = self.unfold(mask.float())
            m_unfold = m_unfold.transpose(1, 2).contiguous().reshape(
                (batch_size, H * W, mask.size(1), -1))  # B*HW*1*9
            m_unfold = m_unfold.transpose(2, 3).contiguous()  # B*HW*9*1

        x_pn_range = torch.norm(x_pn, p=2, dim=-1).unsqueeze(-1)
        x_pn = torch.cat((x_pn, x_pn_range), dim=-1)
        x_p0 = x_pn[:, :, 4:5, :]  # B*HW*1*4
        pn_p0 = x_pn - x_p0  # B*HW*9*4
        pn_p0 = pn_p0.reshape(-1, pn_p0.shape[-1]).contiguous()

        weights_spatial_reduce= self.weight_mlp_spatial(pn_p0[m_unfold.view(-1) > 0])  # N*C'
        weights_channel_reduce = self.weight_mlp_channel(pn_p0[m_unfold.view(-1) > 0])

        weights_spatial = m_unfold.new_zeros(batch_size, H * W, 9, weights_spatial_reduce.shape[-1])
        weights_spatial[m_unfold[..., 0] > 0] = weights_spatial_reduce

        weights_channel = m_unfold.new_zeros(batch_size, H * W, 9, weights_channel_reduce.shape[-1])
        weights_channel[m_unfold[..., 0] > 0] = weights_channel_reduce

        weights_spatial, _ = torch.max(weights_spatial, dim=3, keepdim=True)  # B*HW*9*1'
        weights_channel, _ = torch.max(weights_channel, dim=2, keepdim=True)  # B*HW*1*C'

        weights_spatial = f.softmax(weights_spatial, dim=2)
        weights_channel = f.softmax(weights_channel, dim=3)

        weights = weights_spatial + weights_channel

        # set weights=0 for empty voxels
        if mask is not None and self.use_mask:
            weights = weights * m_unfold

        if self.residual:
            weights = 1 + weights

        features_unfold = weights * features_unfold  # B*HW*9*C'

        features_unfold = features_unfold.reshape(
            (batch_size, -1, 9 * self.in_channels))  # B*HW*9C'

        features_unfold = self.aggregation_mlp(features_unfold)  # B*HW*C''
        features_unfold = features_unfold.transpose(
            1, 2).contiguous()  # B*C''*HW
        features_unfold = self.aggregation_bn(features_unfold)
        features_unfold = self.relu2(features_unfold)
        features = self.fold(features_unfold)

        return features


def main():

    #net = EdgeConvKernel(32, 32, (48, 512), reduced=False,
    #                     use_mask=True).cuda()
    from easydict import EasyDict
    H = 48*2
    W = 512*2
    r = 0.5
    meta_cfg = {
        'META_CHANNELS': 4,
        'INPUT_CHANNELS': 32,
        'OUTPUT_CHANNELS': 32,
        'USE_MASK': True,
        'USE_ATTENTION': True,
        'REDUCED': True,
        'FEATURE_MAP_SIZE': (H, W),
        # 'RESIDUAL': True
            }

    edge_cfg={
        'INPUT_CHANNELS': 32,
        'OUTPUT_CHANNELS': 32,
        'USE_MASK': True,
        'REDUCED': True,
        'FEATURE_MAP_SIZE': (H, W)}

    meta_att_cfg = {
        'META_CHANNELS': 4,
        'INPUT_CHANNELS': 32,
        'OUTPUT_CHANNELS': 32,
        'USE_MASK': True,
        'REDUCED': False,
        'FEATURE_MAP_SIZE': (H, W),
        'RESIDUAL': True
    }
    # net = MetaKernel(EasyDict(meta_cfg)).cuda()
    # net=EdgeConvKernel(EasyDict(edge_cfg)).cuda()
    net=MetaKernelDualAtt(EasyDict(meta_att_cfg)).cuda()

    inp = torch.randn(1, 35, H, W).cuda()
    mask = torch.ones((1, 1, H, W)).cuda().to(torch.bool)
    mask[:, :, :, :int(W * r)] = False
    out = net(inp, mask)
    times = []
    for i in range(100):
        torch.cuda.synchronize()
        t1 = time.time()
        out = net(inp, mask)
        # out2=net(inp,mask,reduced=True)
        # print("out1 equal out2 ",torch.equal(out1,out2))
        # print("diff is ",(out2-out1)[:,0,:10,:10])
        torch.cuda.synchronize()
        t2 = time.time()
        times.append(t2-t1)
    print(np.array(times).mean())
    # w = torch.randn(2, 3, 3, 3)
    # inp_unf = torch.nn.functional.unfold(inp, (3, 3))  # [2,27,6]
    # out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    # out = torch.nn.functional.fold(out_unf, (2, 3), (1, 1))  # [2,2,2,3]
    # print((torch.nn.functional.conv2d(inp, w) - out).abs().max())
    print("out shape is ", out.shape)


if __name__ == '__main__':
    main()
