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

class CrossViewAttention(nn.Module):
    def __init__(self, in_dim, feat_dim=None):
        super(CrossViewAttention, self).__init__()

        out_dim = in_dim // 2 if feat_dim is None else feat_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.att_conv = nn.Conv2d(
            in_channels=out_dim * 2, out_channels=1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the qury feature map
            ref_x (torch.Tensor): the key and value feature map
        """
        batch_size, C, H, W = query_x.size()
        proj_query = self.query_conv(query_x)
        proj_key = self.key_conv(ref_x)
        proj_value = self.value_conv(ref_x)
        
        x = torch.cat([proj_query, proj_key], dim=1)
        x = self.att_conv(x)
        attention = self.sigmoid(x)
        output = query_x + attention * proj_value

        return output

class CrossViewBlockTransformer(nn.Module):
    def __init__(self, query_dim, key_dim, proj_dim, block_size=4, stride=4):
        super(CrossViewBlockTransformer, self).__init__()

        #self.transformer = CrossViewTransformer(query_dim, proj_dim)
        self.transformer = CrossViewTransformerV2(query_dim, key_dim, proj_dim)

        self.block_size = block_size
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=block_size, stride=stride)
        # self.fold = nn.Fold(feature_map_size, kernel_size=block_size, dilation=1, padding=1, stride=stride)

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the query feature map
            ref_x (torch.Tensor): reference feature map
        """
        B, C, H, W = query_x.size()
        
        fold = nn.Fold([H, W], kernel_size=self.block_size, stride=self.stride)

        query_unfold = self.unfold(query_x)  # B * C * H * W ---> B * (C * 4 * 4) * (H2 * W2)
        query_unfold = query_unfold.permute(0, 2, 1)
        query_unfold = query_unfold.reshape(B * query_unfold.shape[1], C, 
                self.block_size, self.block_size).contiguous() # (B * H2 * W2) * C * 4 * 4

        C = ref_x.size(1)
        ref_unfold = self.unfold(ref_x)  # B * C * H * W ---> B * (C * 4 * 4) * (H2 * W2)
        ref_unfold = ref_unfold.permute(0, 2, 1)
        ref_unfold = ref_unfold.reshape(B * ref_unfold.shape[1], C,
                self.block_size, self.block_size).contiguous() # (B * H2 * W2) * C * 4 * 4

        out_unfold = self.transformer(query_unfold, ref_unfold) # (B * H2 * W2) * C2 * 4 * 4
        out_unfold = out_unfold.reshape(out_unfold.shape[0], -1)
        out_unfold = out_unfold.reshape(B, -1, out_unfold.shape[-1]) # B * (H2 * W2) * (C2 * 4 * 4)
        out_unfold = out_unfold.permute(0, 2, 1).contiguous() # B * (C2 * 4 * 4) * (H2 * W2)
        output = fold(out_unfold)

        return output

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

    def forward(self, query_x, ref_x):
        """
        Args:
            query_x (torch.Tensor): the query feature map
            ref_x (torch.Tensor): key and value feature map
        """
        batch_size, C, H, W = ref_x.size()
        proj_query = self.query_conv(query_x).view(
            batch_size, -1, H * W)  # B x C x (N)
        proj_key = self.key_conv(ref_x).view(
            batch_size, -1, H * W).permute(0, 2, 1)  # B x C x (W*H)

        proj_value = self.value_conv(ref_x).view(
            batch_size, -1, H * W)  # B x C x N

        energy = torch.bmm(proj_key, proj_query)  # transpose check

        if self.use_feature_selection:
            ref_star, ref_star_arg = torch.max(energy, dim=1)

            T = feature_selection(proj_value, 2, ref_star_arg).view(
                ref_star.size(0), -1, H, W)

            S = ref_star.view(ref_star.size(0), 1, H, W)

            ref_res = torch.cat((ref_x, T), dim=1)
            ref_res = self.f_conv(ref_res)
            ref_res = ref_res * S
            output = ref_x + ref_res
        else:
            attention = self.softmax(energy)  # B x N x N
            z = torch.bmm(proj_value, attention.permute(0, 2, 1))
            z = z.view(batch_size, C, H, W)
            output = query_x + z

        return output

class CrossViewTransformerV2(nn.Module):
    def __init__(self, query_dim=64, key_dim=64, proj_dim=64, groups=1):
        super().__init__()
        self.groups = groups

        self.query_conv = nn.Conv1d(query_dim, proj_dim, kernel_size=1, stride=1, bias=False)
        self.value_conv = nn.Conv1d(key_dim, proj_dim, kernel_size=1, stride=1, bias=False)
        self.key_conv = nn.Conv1d(key_dim, proj_dim, kernel_size=1, stride=1, bias=False)

        self.fc = nn.Conv1d(proj_dim, query_dim, kernel_size=1, stride=1, groups=self.groups, bias=False)

        # norm (essentially LayerNorm per group)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=query_dim)

        # softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def kernel(self, proj_value, proj_query, proj_key):
        """Return the output after dot product per head
        Args:
            proj_value: output of linear value
            proj_query: output of linear query
            proj_key: output of linear keys (B x C x N)
        """
        # B, C, N = proj_query.shape
        proj_query = proj_query.permute(0, 2, 1)  # B x N x C
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        total_energy = energy # B x N x N
        attention = self.softmax(total_energy)  # B x N x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B x C x N
        return out

    def forward(self, query_x, ref_x):
        residual = query_x
        B, C, H, W = query_x.shape
        query_x = query_x.view(B, -1, H * W)
        ref_x = ref_x.view(B, -1, H * W)

        proj_query = self.query_conv(query_x) # query
        proj_value = self.value_conv(ref_x) # value
        proj_key = self.key_conv(ref_x) # key

        _, C2, N = proj_value.size()

        if self.groups and self.groups > 1:
            _c = int(C2 / self.groups)

            _proj_value = torch.split(proj_value, split_size_or_sections=_c, dim=1)
            _proj_query = torch.split(proj_query, split_size_or_sections=_c, dim=1)
            _proj_key = torch.split(proj_key, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(_proj_value[i], _proj_query[i], _proj_key[i])
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(proj_value, proj_query, proj_key)

        x = self.fc(x) # B x C x N
        x = self.gn(x) # 1 x C  x N
        x = x.reshape(B, C, H, W)
        out = x + residual
        return out 

if __name__ == '__main__':
    
    dim = 64
    H, W = 248 // 2, 216 // 2
    rv_x = torch.rand(4, dim, H, W)
    bev_x = torch.rand(4, dim, H, W)
    #cross_view = CrossViewTransformer(dim, use_feature_selection=False)
    cross_view = CrossViewAttention(dim)
    out = cross_view(rv_x, bev_x)
    print(out.shape)
