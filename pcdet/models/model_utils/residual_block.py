"""Residual block."""

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block.

    Args:
        input_channels (int): Input channels.
        output_channels (int): Output channels.
    """

    def __init__(self, input_channels, output_channels, stride=1, use_relu=True):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, stride=1, bias = False)
        self.bn1 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias = False)

    def forward(self, x):
        """Forward function.
        
        Args:
            x (torch.Tensor): Input with shape (N, c, H, W)

        Returns:
            torch.Tensor: residual output.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(x)

        out += residual
        if self.use_relu:
            out = self.relu(out)

        return out


def test():
    """test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    x = torch.randn(3, 32, 31, 27).float().to(device)
    
    print(x.shape)

    residual = ResidualBlock(32, 64).to(device)
    r = residual(x)
    print(r.shape)


if __name__ == '__main__':
    test()