from ..backbones_2d import map_to_bev
from ..backbones_3d import vfe
from .. import backbones_2d, dense_heads
from .detector3d_template import Detector3DTemplate


class RangeDet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.build_networks()

    def forward(self, batch_dict):
        if self.vfe:
            batch_dict = self.vfe(batch_dict)

        if self.map_to_bev:
            batch_dict = self.map_to_bev(batch_dict)

        batch_dict = self.backbone_fv(batch_dict)
        batch_dict = self.range_to_bev(batch_dict)
        batch_dict = self.backbone_bev(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.model_cfg.DENSE_HEAD.NAME=="CenterHead":
                pred_dicts, recall_dicts = self.post_processingV2(batch_dict)
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'depth_downsample_factor': self.dataset.depth_downsample_factor,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }
        
        self.vfe, model_info_dict = self.build_vfe(model_info_dict)
        self.map_to_bev, model_info_dict = self.build_map_to_bev_module(model_info_dict)

        self.backbone_fv, model_info_dict = self.build_backbone_fv(model_info_dict=model_info_dict)
        self.range_to_bev, model_info_dict = self.build_range_to_bev(model_info_dict=model_info_dict)
        self.backbone_bev, model_info_dict = self.build_backbone_bev(model_info_dict=model_info_dict)
        self.dense_head, model_info_dict = self.build_dense_head(model_info_dict=model_info_dict)
    
    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict
    
    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_fv(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_FV', None) is None:
            return None, model_info_dict

        backbone_fv_module = backbones_2d.__all__[self.model_cfg.BACKBONE_FV.NAME](
            in_channels=self.model_cfg.BACKBONE_FV.INPUT_CHANNELS,
            out_channels=self.model_cfg.BACKBONE_FV.OUTPUT_CHANNELS,
            append_far=self.model_cfg.BACKBONE_FV.get('APPEND_FAR',False),
            kernel_cfg=self.model_cfg.BACKBONE_FV.get('KERNEL_CFG', None)
        )

        model_info_dict['module_list'].append(backbone_fv_module)
        return backbone_fv_module, model_info_dict

    def build_backbone_bev(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_BEV', None) is None:
            return None, model_info_dict

        backbone_bev_module = backbones_2d.__all__[self.model_cfg.BACKBONE_BEV.NAME](
            model_cfg=self.model_cfg.BACKBONE_BEV,
            input_channels=self.model_cfg.BACKBONE_BEV.INPUT_CHANNELS
        )

        model_info_dict['module_list'].append(backbone_bev_module)
        return backbone_bev_module, model_info_dict
    
    def build_range_to_bev(self, model_info_dict):
        if self.model_cfg.get('RANGE_TO_BEV', None) is None:
            return None, model_info_dict

        range_to_bev_module = backbones_2d.__all__[self.model_cfg.RANGE_TO_BEV.NAME](
            model_cfg = self.model_cfg.RANGE_TO_BEV)

        model_info_dict['module_list'].append(range_to_bev_module)
        return range_to_bev_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        if self.model_cfg.DENSE_HEAD.NAME=='CenterHead':
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=self.model_cfg.DENSE_HEAD.INPUT_CHANNELS,
                tasks=self.model_cfg.DENSE_HEAD.TASKS,
                train_cfg=self.model_cfg.DENSE_HEAD.TRAIN_CONFIG,
                bbox_coder=self.model_cfg.DENSE_HEAD.BBOX_CODER,
                common_heads=self.model_cfg.DENSE_HEAD.COMMON_HEADS,
                separate_head=self.model_cfg.DENSE_HEAD.SEPARATE_HEAD,
                share_conv_channel=self.model_cfg.DENSE_HEAD.SHARE_CONV_CHANNEL,)
        else:
            dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
                model_cfg=self.model_cfg.DENSE_HEAD,
                input_channels=self.model_cfg.DENSE_HEAD.INPUT_CHANNELS,
                num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
                class_names=self.class_names,
                grid_size=model_info_dict['grid_size'],
                point_cloud_range=model_info_dict['point_cloud_range'],
                predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
            )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
