import numpy as np
import torch
import torch.nn as nn

from pcdet.models.model_utils.generate_pillar_mask import GeneragePillarMask

"""
We introduce two different BEV decoders - ConcatBEV and ConcatVoxel Decoders.
These decoders, unlike the BaseBEVDecoder concatenates the convolutional
feature map from the encoder, with the self-attention features obtained
after the cfe module operation.
"""


class BaseBEVDecoder(nn.Module): 
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_name = model_cfg.get('FEATRUE_NAME', 'spatial_features')

        if self.model_cfg.get('NUM_FILTERS', None) is not None:
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            num_filters = []
        self.num_levels = len(num_filters)

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(inplace=True)
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_name, i)]

            x = data_dict['{}_{}x'.format(self.feature_name, stride)]

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > self.num_levels:
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
