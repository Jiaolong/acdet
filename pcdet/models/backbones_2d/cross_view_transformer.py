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
    def __init__(self, query_dim=64, key_dim=64, proj_dim=8, groups=1):
        super(CrossViewTransformer, self).__init__()

        self.query_conv = nn.Conv2d(
            in_channels=query_dim, out_channels=proj_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=key_dim, out_channels=proj_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=key_dim, out_channels=query_dim, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the query feature map
            ref_x (torch.Tensor): the reference feature map
        """
        batch_size, C, H, W = ref_x.size()
        proj_query = self.query_conv(query_x).view(
            batch_size, -1, H * W)  # B x C x N
        proj_key = self.key_conv(ref_x).view(
            batch_size, -1, H * W).permute(0, 2, 1).contiguous()  # B x N x C

        proj_value = self.value_conv(ref_x).view(
            batch_size, -1, H * W)  # B x C x N

        energy = torch.bmm(proj_key, proj_query)  # transpose check  B*N*N

        attention = self.softmax(energy)  # B x N x N
        z = torch.bmm(proj_value, attention.permute(0, 2, 1).contiguous())
        z = z.view(batch_size, C, H, W)
        output = query_x + z

        return output
