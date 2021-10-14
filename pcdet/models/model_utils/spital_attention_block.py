import numpy as np
import torch

from torch import nn as nn
from pcdet.models.model_utils.scale_aware_block import EncoderBlock

class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()
    def forward(self, x):
        if isinstance(x, list):
            return torch.cat((torch.max(x[0],1)[0].unsqueeze(1),
                              torch.mean(x[0],1).unsqueeze(1),
                              torch.max(x[1], 1)[0].unsqueeze(1),
                              torch.mean(x[1], 1).unsqueeze(1)), dim=1)
        else:
            return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialAttention(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=[128, 128, 128],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 ):
        super(SpatialAttention, self).__init__()
        self.encoder_block = EncoderBlock(in_channels, out_channels, layer_nums,
                                          layer_strides)
        spatials = []
        for i in range(len(out_channels)):
            spatial = [nn.Conv2d(in_channels=out_channels[i], out_channels=2,
                                kernel_size=3, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(num_features=2, eps=1e-3, momentum=0.01),
                       nn.ReLU()]
            spatial.append(nn.Conv2d(in_channels=2, out_channels=1,
                                 kernel_size=3, stride=1, padding=1))
            spatial.append(nn.Sigmoid())

            spitals = nn.Sequential(*spatial)
            spatials.append(spitals)

        self.spatials = nn.ModuleList(spatials)

    def forward(self, x, context=None):
        x = self.encoder_block(x)
        if context is None: out = [self.spatials[i](x[i]) for i in range(len(x))]
        else: out = [self.spatials[i]([x[i], context[i]]) for i in range(len(x))]
        return out