import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_selection(input, dim, index):
    views = [input.size(0)] + [1 if i != dim else -
                               1 for i in range(1, len(input.size()))]
    expanse = list(input.size())
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class CrossViewTransformer(nn.Module):
    def __init__(self, in_dim, feat_dim=None, use_feature_selection=False):
        super(CrossViewTransformer, self).__init__()
        self.use_feature_selection = use_feature_selection

        out_dim = in_dim // 8 if feat_dim is None else feat_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        if self.use_feature_selection:
            self.f_conv = nn.Conv2d(in_channels=in_dim * 2, out_channels=in_dim,
                                    kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, rv_x, bev_x):
        """
        Args:
            rv_x (torch.Tensor): range view feature map
            bev_x (torch.Tensor): BEV feature map, they query feature
        """
        batch_size, C, H, W = rv_x.size()
        proj_query = self.query_conv(bev_x).view(
            batch_size, -1, H * W)  # B x C x (N)
        proj_key = self.key_conv(rv_x).view(
            batch_size, -1, H * W).permute(0, 2, 1)  # B x C x (W*H)

        proj_value = self.value_conv(rv_x).view(
            batch_size, -1, H * W)  # B x C x N

        energy = torch.bmm(proj_key, proj_query)  # transpose check

        if self.use_feature_selection:
            rv_star, rv_star_arg = torch.max(energy, dim=1)

            T = feature_selection(proj_value, 2, rv_star_arg).view(
                rv_star.size(0), -1, H, W)

            S = rv_star.view(rv_star.size(0), 1, H, W)

            rv_res = torch.cat((rv_x, T), dim=1)
            rv_res = self.f_conv(rv_res)
            rv_res = rv_res * S
            output = rv_x + rv_res
        else:
            attention = self.softmax(energy)  # B x N x N
            z = torch.bmm(proj_value, attention.permute(0, 2, 1))
            z = z.view(batch_size, C, H, W)
            output = bev_x + z

        return output


if __name__ == '__main__':
    
    dim = 64
    H, W = 248 // 2, 216 // 2
    rv_x = torch.rand(4, dim, H, W)
    bev_x = torch.rand(4, dim, H, W)
    cross_view = CrossViewTransformer(dim, use_feature_selection=False)
    out = cross_view(rv_x, bev_x)
    print(out.shape)
