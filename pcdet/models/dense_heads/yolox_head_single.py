import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from .anchor_head_template import AnchorHeadTemplate


class YOLOXHeadSingle(AnchorHeadTemplate):
    def __init__(self,
            model_cfg,
            input_channels,
            num_class,
            class_names,
            grid_size,
            point_cloud_range,
            predict_boxes_when_training=True,
            feat_channels=128,
            stacked_convs=2,
            **kwargs):
        super().__init__(
                model_cfg=model_cfg,
                num_class=num_class,
                class_names=class_names,
                grid_size=grid_size,
                point_cloud_range=point_cloud_range,
                predict_boxes_when_training=predict_boxes_when_training)

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.input_channels = input_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs

        self._init_layers()
        self._init_weights()

    def _init_layers(self):

        self.stacked_convs_cls = self._build_stacked_convs()
        self.stacked_convs_reg = self._build_stacked_convs()

        self._build_predictor()

    def _build_stacked_convs(self):
        if self.stacked_convs <= 0:
            return None

        stacked_convs = []

        for i in range(self.stacked_convs):
            chn = self.input_channels if i == 0 else self.feat_channels
            stacked_convs.extend([
                nn.Conv2d(chn, self.feat_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.feat_channels, eps=0.001, momentum=0.03),
                nn.ReLU()
                ])
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        self.predictor_cls = nn.Conv2d(self.feat_channels,
                self.num_anchors_per_location *
                self.num_class,
                kernel_size=1)
        self.predictor_box = nn.Conv2d(self.feat_channels,
                self.num_anchors_per_location *
                self.box_coder.code_size,
                kernel_size=1)
        self.predictor_obj = nn.Conv2d(self.feat_channels,
                self.num_anchors_per_location,
                kernel_size=1)
        self.predictor_dir_cls = nn.Conv2d(self.feat_channels,
                self.num_anchors_per_location *
                self.model_cfg.NUM_DIR_BINS,
                kernel_size=1)

    def _init_weights(self):
        pi = 0.01
        nn.init.constant_(self.predictor_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.predictor_box.weight, mean=0, std=0.001)

    def build_losses(self, losses_cfg):
        self.add_module(
                'cls_loss_func',
                #loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
                loss_utils.WeightedCrossEntropyLoss(use_sigmoid=True)
                )
        self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
                )
        self.add_module(
                'obj_loss_func',
                loss_utils.WeightedCrossEntropyLoss(use_sigmoid=True)
                )
        self.add_module(
                'dir_loss_func',
                loss_utils.WeightedCrossEntropyLoss()
                )
    
    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        batch_size = int(cls_preds.shape[0])
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # (B, N)
        box_iou_targets = self.forward_ret_dict['box_iou_targets'] # (B, N)
        
        cls_weights = (box_cls_labels > 0).float()
        pos_normalizer = cls_weights.sum(1, keepdim=True).float() # (B, N)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_labels = box_cls_labels - 1
        cls_labels[cls_labels < 0] = 0
        cls_targets = F.one_hot(cls_labels, self.num_class) * box_iou_targets.unsqueeze(-1) 
        
        cls_loss_src = self.cls_loss_func(cls_preds, cls_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict
    
    def get_obj_layer_loss(self):
        obj_preds = self.forward_ret_dict['obj_preds'] # (B, H, W, 1)
        batch_size = int(obj_preds.shape[0])
        obj_preds = obj_preds.view(batch_size, -1, 1) # (B, N, 1)
        
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # (B, N)
        pos_mask = box_cls_labels > 0

        obj_targets = torch.zeros_like(obj_preds) # (B, N, 1)
        obj_targets[pos_mask] = 1

        obj_weights = torch.ones_like(obj_preds[..., 0]) # (B, N)
        pos_normalizer = pos_mask.sum(1, keepdim=True).float() # (B, N)
        obj_weights /= torch.clamp(pos_normalizer, min=1.0)

        obj_loss = self.obj_loss_func(obj_preds, obj_targets, weights=obj_weights)
        obj_loss = obj_loss.sum() / batch_size
        obj_loss = obj_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['obj_weight']
        tb_dict = {
                'rpn_loss_obj': obj_loss.item()
        }
        return obj_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()

        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        
        obj_loss, tb_dict_obj = self.get_obj_layer_loss()
        tb_dict.update(tb_dict_obj)

        rpn_loss = cls_loss + box_loss + obj_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, obj_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            obj_preds: (N, H, W, 1)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                    for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors

        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
                if not isinstance(cls_preds, list) else cls_preds
        batch_obj_preds = obj_preds.view(batch_size, num_anchors, -1).float()
        batch_cls_preds = batch_cls_preds.sigmoid() * batch_obj_preds.sigmoid()

        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
                else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                    else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                    )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                    -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
                    )

        return batch_cls_preds, batch_box_preds

    def assign_targets(self, gt_boxes, cls_preds, box_preds):
        """
        Args:
            gt_boxes: (B, M, 8)
            cls_preds: (B, N, num_class)
            box_preds: (B, N, 7)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
                self.anchors, gt_boxes, batch_cls_preds=cls_preds, batch_box_preds=box_preds
                )
        return targets_dict

    def forward(self, data_dict):
        x = data_dict['spatial_features_2d']
        
        if self.stacked_convs > 0:
            cls_feat = self.stacked_convs_cls(x)
            reg_feat = self.stacked_convs_reg(x)
        else:
            cls_feat, reg_feat = x, x

        cls_preds = self.predictor_cls(cls_feat)
        box_preds = self.predictor_box(reg_feat)
        obj_preds = self.predictor_obj(reg_feat)
        dir_cls_preds = self.predictor_dir_cls(reg_feat)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        obj_preds = obj_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['obj_preds'] = obj_preds
        self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds

        # get decoded boxes
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds,
                box_preds=box_preds,
                obj_preds=obj_preds,
                dir_cls_preds=dir_cls_preds)

        if self.training:
            targets_dict = self.assign_targets(gt_boxes=data_dict['gt_boxes'],
                    cls_preds=batch_cls_preds, box_preds=batch_box_preds)
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
