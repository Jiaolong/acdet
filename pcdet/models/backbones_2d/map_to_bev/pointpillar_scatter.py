import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class PointPillarScatter_Scale(nn.Module):
    def __init__(self,
                 model_cfg, grid_size, **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        Masked version
        """

        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_scale_features = self.model_cfg.NUM_SCALE_FEATURES
        self.nx, self.ny, self.nz = grid_size
        
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, pillar_scale_features, coords = (
                batch_dict['pillar_features'], 
                batch_dict['pillar_scale_features'],
                batch_dict['voxel_coords'])

        batch_spatial_features = []
        batch_spatial_scale_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            spatial_scale_feature = torch.zeros(
                self.num_scale_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_scale_features.dtype,
                device=pillar_scale_features.device)
 
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()

            pillars_scale = pillar_scale_features[batch_mask, :]
            pillars_scale = pillars_scale.t()
           
            spatial_feature[:, indices] = pillars
            spatial_scale_feature[:, indices] = pillars_scale
     
            batch_spatial_features.append(spatial_feature)
            batch_spatial_scale_features.append(spatial_scale_feature)
   
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_scale_features = torch.stack(batch_spatial_scale_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_scale_features = batch_spatial_scale_features.view(batch_size, self.num_scale_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        batch_dict['spatial_scale_features'] = batch_spatial_scale_features
        
        return batch_dict
