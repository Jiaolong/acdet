import numpy as np
import torch
import torch.nn as nn


"""
This is the Encoder in the BaseBackbone2d.
We separate it in order to be able to access the 
encoder features directly.
"""


class BaseBEVEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_name = model_cfg.get('FEATURE_NAME', 'spatial_features')

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict[self.feature_name]
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict['{}_{}x'.format(self.feature_name, stride)] = x
            data_dict['{}_stride_{}'.format(self.feature_name, i)] = stride
        return data_dict

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters,
                               (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class DownsampleBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters,
                               kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), stride=2, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)
        output = self.bn1(x)

        return output

class RawBEVEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_name = model_cfg.get('FEATURE_NAME', 'points_img_bev')
        self.feat_channels = model_cfg.get('FEAT_CHANNELS', 32)
        self.downsample_input = model_cfg.get('DOWNSAMPLE_INPUT', False)
        
        if self.downsample_input:
            self.conv1 = DownsampleBlock(input_channels, self.feat_channels)
        else:
            self.conv1 = ResContextBlock(input_channels, self.feat_channels)

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        num_levels = len(layer_nums)
        c_in_list = [self.feat_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict[self.feature_name]
        
        x = self.conv1(spatial_features)
        H = x.shape[2]

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(H / x.shape[2])
            data_dict['{}_{}x'.format(self.feature_name, stride)] = x
            data_dict['{}_stride_{}'.format(self.feature_name, i)] = stride
        return data_dict

class MaskBlock(nn.Module):
    def __init__(self, input_channels, num_filters, stride):
        super().__init__()
        
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(input_channels, num_filters, 
                stride=stride, kernel_size=3, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.01)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        self.act2 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # channel-wise max pooling
        x1 = x.max(dim=1, keepdim=True).values
        
        # channel-wise average pooling
        x2 = x.mean(dim=1, keepdim=True)

        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.conv2(x3)
        x3 = self.bn2(x3)

        out = self.act2(x3)
        return out

class MaskBEVEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_name = model_cfg.get('FEATURE_NAME', 'spatial_features')

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.blocks = nn.ModuleList()
        self.mask_blocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))

            self.mask_blocks.append(MaskBlock(c_in_list[idx], num_filters[idx], stride=layer_strides[idx]))

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict[self.feature_name]
        x = spatial_features

        for i in range(len(self.mask_blocks)):
            m = self.mask_blocks[i](x)
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            data_dict['{}_{}m'.format(self.feature_name, stride)] = m
            data_dict['{}_stride_{}'.format(self.feature_name, i)] = stride
            data_dict['{}_{}x'.format(self.feature_name, stride)] = x
        return data_dict

