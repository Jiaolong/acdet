import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pcdet.datasets.processor.projection import BEVProjector
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils
class RangeToBEV(nn.Module):
    """Convert range/cylinder view projected features to BEV features.


    Args:
        project_cfg (dict): bev projection configurations
        with_raw_features (bool): if True, concatenate raw point feature (x, y, z, intensity) with projected features
    """

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        project_cfg = self.model_cfg.PROJECT_CFG
        self.with_raw_features = False
        self.voxel_size = torch.tensor(project_cfg['VOXEL_SIZE'], dtype=torch.float).cuda().reshape(-1, 3)
        self.point_cloud_range = torch.tensor(project_cfg['POINT_CLOUD_RANGE'], dtype=torch.float).cuda().reshape(-1, 6)
        self.bev_projector_range = BEVProjector(project_cfg)

    def forward(self,data_dict):
        return  self.forward_v1(data_dict)

    def forward_v1(self, data_dict):
        """Foraward function to projected features.

        Args:
            x (tensor): range/cylinder view projected features (B, C, H, W)
            points_img (tensor): range/cylinder view projected points (B, 4, H, W)
            proj_masks (tensor): range/cylinder view projection mask (B, H, W)
        Returns:
            bev_features_batch (tensor): BEV projected features (B, C, H2, W2) 
        """

        x = data_dict['fv_features']
        points_img = data_dict['points_img']
        proj_masks = data_dict['proj_masks']
        batch_size, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        features_batch = x.view(batch_size, H * W, C)  # B, H * W, C
        points_batch = points_img[:, :3].permute(0, 2, 3, 1).contiguous()  # B, H, W, 3
        points_batch = points_batch.view(batch_size, H * W, -1)  # B, H * W, 3
        proj_masks = proj_masks.view(batch_size,H*W)

        bev_features_batch = []
        for k in range(batch_size):
            points = points_batch[k]  # H * W, 4
            features = features_batch[k]  # H * W, C
            mask = proj_masks[k]  # H * W
            points = points[mask > 0, :]
            features = features[mask > 0, :]

            # project points to BEV
            bev_features = self.bev_projector_range.get_projected_features(points, features, self.with_raw_features)  # C + 4, H2, W2
            bev_features_batch.append(bev_features)
        bev_features_batch = torch.stack(bev_features_batch,dim=0)  # B, C + 4, H2, W2

        data_dict['spatial_features'] = bev_features_batch
        return data_dict

    def forward_bilinear(self, x, points_img, proj_masks, points_with_proj_coords):
        """Foraward function to projected features.

        Bilinear interpolation is used to aggregate point-wise features.

        Args:
            x (tensor): range/cylinder view projected features (B, C, H, W)
            points_img (tensor): range/cylinder view projected points (B, 4, H, W)
            proj_masks (tensor): range/cylinder view projection mask (B, H, W)
            points_with_proj_coords (tensor): points with projection coordinates merged in all batches (N, 7)
        Returns:
            bev_features_batch (tensor): BEV projected features (B, C, H2, W2) 
        """
        batch_size, C, H, W = x.shape

        batch_inds = points_with_proj_coords[:, 0]
        points_batch = points_with_proj_coords[:, 1:5]
        # (N, 2) proj_x, proj_y in feature map coordinates
        coords_batch = points_with_proj_coords[:, 5:7]
        # normalize to grid coordinate [-1, 1]
        coords_batch[:, 0] = 2.0 * (coords_batch[:, 0] / W) - 1.0  # proj_x
        coords_batch[:, 1] = 2.0 * (coords_batch[:, 1] / H) - 1.0  # proj_y

        bev_features_batch = []
        for k in range(batch_size):
            points = points_batch[batch_inds == k]  # npoints, 4
            features_raw = self.pointnet_raw(points)  # npoints, C1

            coords = coords_batch[batch_inds == k]
            coords = coords.view(1, 1, -1, 2)  # 1, 1, npoints, 2

            features = features_batch[k:k+1]  # 1, C, H, W
            features_sampled = F.grid_sample(
                features, coords, mode='bilinear', align_corners=False)  # 1, C, 1, npoints
            features_sampled = features_sampled.squeeze()  # C, npoints
            features_sampled = features_sampled.permute(1, 0)  # npoints, C
            features_fused = torch.cat(
                [features_raw, features_sampled], dim=1)  # npoints, C + C1

            features_fused = self.pointnet_fuse(features_fused)  # npoints, C2

            # project points to BEV
            bev_features = self.bev_projector.get_projected_features(
                points, features_fused)  # C2, H2, W2

            bev_features_batch.append(bev_features)

        bev_features_batch = torch.stack(bev_features_batch)  # B, C2, H2, W2

        return bev_features_batch

    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points ([torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        batch_size=points[-1,0].item()+1
        for i in range(int(batch_size)):
            cur_mask=points[:,0]==i
            cur_points=points[cur_mask][:,1:].contiguous()
            cur_coors = self.voxel_layer(cur_points)
            coor_pad = F.pad(cur_coors, (1, 0), mode='constant', value=i)
            coors.append(coor_pad)
        coors_batch = torch.cat(coors, dim=0)
        return coors_batch

    @torch.no_grad()
    def voxelize_v2(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points ([torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        batch_size = points[-1, 0].item() + 1
        for i in range(int(batch_size)):
            cur_mask = points[:, 0] == i
            cur_points = points[cur_mask]
            cur_coors = ((cur_points[:, [2, 1, 0]] - self.point_cloud_range[:,[2,1,0]]) / self.voxel_size[:,[2, 1, 0]]).to(torch.int64)
            coor_pad = F.pad(cur_coors, (1, 0), mode='constant', value=i)
            coors.append(coor_pad)
        coors_batch = torch.cat(coors, dim=0)
        return coors_batch

