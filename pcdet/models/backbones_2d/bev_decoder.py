import numpy as np
import torch
import torch.nn as nn

from .cross_view_transformer import CrossViewTransformer
from .cross_view_transformer import CrossViewAttention
from .cross_view_transformer import CrossViewBlockTransformer
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
                        nn.ReLU()
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
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
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


class ConcatBEVDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
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
                            input_channels + num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            input_channels + num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
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
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)]
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)]
            x = torch.cat([x1, x2], dim=1)

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

class LateConcatBEVDecoder(nn.Module):
    """
    Concat the output of two individual decoder branches.
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
        self.with_transformer = model_cfg.get('WITH_TRANSFORMER', False)
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
        
        self.deblocks_1 = nn.ModuleList()
        self.deblocks_2 = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks_1.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks_1.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.deblocks_2.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks_1.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.deblocks_2.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in
        if self.with_transformer:
            self.transformer = CrossViewBlockTransformer(in_dim=c_in, block_size=4, stride=4)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups_1 = []
        ups_2 = []

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)]
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)]

            if len(self.deblocks_1) > 0:
                ups_1.append(self.deblocks_1[i](x1))
            else:
                ups_1.append(x1)

            if len(self.deblocks_2) > 0:
                ups_2.append(self.deblocks_2[i](x2))
            else:
                ups_2.append(x2)

        if len(ups_1) > 1:
            x1 = torch.cat(ups_1, dim=1)
        elif len(ups_1) == 1:
            x1 = ups_1[0]

        if len(ups_2) > 1:
            x2 = torch.cat(ups_2, dim=1)
        elif len(ups_2) == 1:
            x2 = ups_2[0]

        if len(self.deblocks_1) > self.num_levels:
            x1 = self.deblocks_1[-1](x1)

        if len(self.deblocks_2) > self.num_levels:
            x2 = self.deblocks_2[-1](x1)

        
        if self.with_transformer:
            x = self.transformer(x1, x2)
        else:
            x = torch.cat([x1, x2], dim=1)

        data_dict['spatial_features_2d'] = x
        return data_dict

class CrossViewTransformerBEVDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
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
                            #input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            # input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in
        
        self.transformer = CrossViewTransformer(query_dim=input_channels, key_dim=input_channels, proj_dim=input_channels // 8)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)] # range view
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # bev view
            if i == self.num_levels - 1:
                x = self.transformer(x1, x2)
            else:
                # x = torch.cat([x1, x2], dim=1)
                x = x1

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

class CrossViewBlockTransformerBEVDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
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
                            #input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            # input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in
        
        # global transformer for last layer
        # self.transformer = CrossViewTransformer(in_dim=input_channels)

        self.transformers = nn.ModuleList()
        for idx in range(self.num_levels):
            # block-wise transformer for high resolution feature maps
            transformer = CrossViewBlockTransformer(
                    query_dim=input_channels, key_dim=input_channels, proj_dim=input_channels, block_size=4, stride=4)
            self.transformers.append(transformer)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)] # range view
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # bev view
            x = self.transformers[i](x1, x2)
            #if i == self.num_levels - 1:
            #    x = self.transformer(x1, x2)
            #else:
            #    x = self.block_transformer(x1, x2)

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

class CrossViewAttentionBEVDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
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
        self.attention_blocks = nn.ModuleList()

        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                
                self.attention_blocks.append(
                        CrossViewAttention(in_dim=input_channels)
                        )

                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            input_channels + num_filters[idx], num_upsample_filters[idx],
                            #num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            input_channels + num_filters[idx], num_upsample_filters[idx],
                            #num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
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
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)] # range view
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # bev view
            x = self.attention_blocks[i](x1, x2)
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

class ConcatPillarCtxDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
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
                            self.model_cfg.IN_DIM + num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            self.model_cfg.IN_DIM + num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features

        for i in range(self.num_levels):
            stride = data_dict['spatial_features_stride_{}'.format(i)]
            x = data_dict['spatial_features_%dx' % stride]
            x = torch.cat([x, data_dict['pillar_context'][i]], dim=1)

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

class CrossViewTransformerMaskBEVDecoder(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
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
        
        self.voxel_size = [0.16, 0.16, 4]
        self.point_clout_range = [0, -39.68, -3, 69.12, 39.68, 1]
        self.forward_ret_dict = {}

        self.deblocks = nn.ModuleList()
        for idx in range(self.num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            #input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            # input_channels + num_filters[idx], num_upsample_filters[idx],
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.binary_cls = nn.Sequential(
            nn.Conv2d(c_in, num_upsample_filters[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(num_upsample_filters[0], num_upsample_filters[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(num_upsample_filters[0], 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.num_bev_features = c_in

        self.transformer = CrossViewTransformer(query_dim=input_channels, key_dim=input_channels, proj_dim=input_channels // 8)
        self.mask_generate = GeneragePillarMask(voxel_size=self.voxel_size, point_cloud_range=self.point_clout_range)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)] # range view
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # bev view
            if i == self.num_levels - 1:
                x = self.transformer(x1, x2)
            else:
                # x = torch.cat([x1, x2], dim=1)
                x = x1

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

        mask = self.binary_cls(x)
        gt_mask = self.mask_generate.generate(mask, data_dict["points"], data_dict['voxel_coords'], 
                                    self.voxel_size, self.point_clout_range, data_dict["gt_boxes"])
        self.forward_ret_dict["mask"] = mask
        self.forward_ret_dict["gt_mask"] = gt_mask.to(mask.device).unsqueeze(1)
        data_dict['spatial_features_2d'] = torch.mul(x, mask) + x
        return data_dict

    def get_loss(self):
        prediction = self.forward_ret_dict["mask"]
        target = self.forward_ret_dict["gt_mask"]
        tb_dict = dict()
        self.alpha = 2
        self.beta = 4
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.
        # prediction = torch.clamp(prediction, 1e-3, .999)
        positive_loss = torch.log(prediction + 1e-6) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction + 1e-6) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive
        tb_dict["loss_heatmap"] = loss.item()

        # dice loss
        # intersection = (target * prediction).sum(axis=[1,2,3])
        # dice_score = (2 * intersection + 1) / (target.sum(axis=[1,2,3]) + prediction.sum(axis=[1,2,3]) + 1)
        # dice_loss = 1 - torch.mean(dice_score, axis=0)
        # loss_dict["loss_dice"] = dice_loss * 0.2
        # if torch.isnan(loss) or torch.isnan(dice_loss):
        #     import pdb;pdb.set_trace()

        return loss, tb_dict
