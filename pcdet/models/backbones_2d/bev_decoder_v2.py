import numpy as np
import torch
import torch.nn as nn

from .bev_decoder import BaseBEVDecoder
from .cross_view_transformer import CrossViewTransformer
from pcdet.models.model_utils.generate_pillar_mask import GeneragePillarMask

class CrossViewTransformerMaskBEVDecoderV2(BaseBEVDecoder):
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels)
        self.model_cfg = model_cfg
        self.feature_names = model_cfg.get('FEATURE_NAMES', [])
        self.multi_level=model_cfg.get('MULTI_LEVEL',False)
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
        
        self.voxel_size = self.model_cfg.get('VOXEL_SIZE', None)
        self.point_clout_range = self.model_cfg.get('POINT_CLOUD_RANGE', None)
        self.use_mask_supervision = self.model_cfg.get('USE_MASK_GT', True)
        self.use_transformer = self.model_cfg.get('USE_TRANSFORMER', True)

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
                        nn.ReLU(inplace=True)
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
                        nn.ReLU(inplace=True)
                    ))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > self.num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ))
        
        mask_in_dim = num_filters[-1]
        mask_filters = num_upsample_filters[-1]
        self.mask_net = nn.Sequential(
            # upsample
            nn.ConvTranspose2d(
                mask_in_dim, mask_filters,
                upsample_strides[-1],
                stride=upsample_strides[-1], bias=False
            ),
            nn.BatchNorm2d(mask_filters, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),

            # conv blocks
            nn.Conv2d(mask_filters, mask_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(mask_filters, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_filters, mask_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(mask_filters, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_filters, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.num_bev_features = c_in
        
        if self.use_transformer:
            self.transformer = CrossViewTransformer(query_dim=input_channels, key_dim=input_channels, proj_dim=input_channels // 8)

        if self.use_mask_supervision:
            self.mask_generate = GeneragePillarMask(voxel_size=self.voxel_size, point_cloud_range=self.point_clout_range)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        ups = []
        
        # predict mask
        stride = data_dict['{}_stride_{}'.format(self.feature_names[1], self.num_levels - 1)]
        x = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # features in bev view
        mask = self.mask_net(x)
        self.forward_ret_dict["mask"] = mask

        if self.training and self.use_mask_supervision:
            gt_mask = self.mask_generate.generate(mask, data_dict["points"], data_dict['voxel_coords'],
                                    self.voxel_size, self.point_clout_range, data_dict["gt_boxes"])
            self.forward_ret_dict["gt_mask"] = gt_mask.to(mask.device).unsqueeze(1)

        for i in range(self.num_levels):
            stride = data_dict['{}_stride_{}'.format(self.feature_names[0], i)]

            x1 = data_dict['{}_{}x'.format(self.feature_names[0], stride)] # range view
            x2 = data_dict['{}_{}x'.format(self.feature_names[1], stride)] # bev view
            if i == self.num_levels - 1:
                if self.use_transformer:
                    x = self.transformer(x1, x2)
                else:
                    x = x1 + x2 # TODO: concat or other fusion methods
            else:
                # x = x1+x2
                if self.multi_level:
                    x=x1+x2
                else:
                    x=x1

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
        
        data_dict['spatial_features_2d'] = torch.mul(x, mask) + x
        return data_dict

    def get_loss(self):
        if not self.use_mask_supervision:
            return None, None

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

        return loss, tb_dict

