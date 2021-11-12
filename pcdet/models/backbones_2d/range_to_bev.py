import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pcdet.datasets.processor.projection import BEVProjector
from pcdet.models.backbones_2d.meta_kernel import EdgeConvKernel, MetaKernel
from pcdet.ops.voxel import Voxelization,DynamicScatter,DynamicVoxelization
from pcdet.models.backbones_2d.map_to_bev import PointPillarScatter
from easydict import EasyDict
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
        self.use_voxelization=model_cfg.get('VOXELIZATION',False)
        self.fuse_complete_points = self.model_cfg.get('FUSE_COMPLETE_POINTS', False)
        self.with_raw_features = False
        self.voxel_size = torch.tensor(project_cfg['VOXEL_SIZE'], dtype=torch.float).cuda().reshape(-1, 3)
        self.point_cloud_range = torch.tensor(project_cfg['POINT_CLOUD_RANGE'], dtype=torch.float).cuda().reshape(-1, 6)


        if self.use_voxelization:
            self.voxel_layer = Voxelization(voxel_size=project_cfg['VOXEL_SIZE'],
                                            point_cloud_range=project_cfg['POINT_CLOUD_RANGE'],
                                            max_num_points=-1,
                                            max_voxels=-1, )
            point_cloud_range = np.array(project_cfg['POINT_CLOUD_RANGE'])
            voxel_size = np.array(project_cfg['VOXEL_SIZE'])
            self.grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
            self.grid_size = np.round(self.grid_size).astype(np.int64)
            pillar_scatter_cfg = EasyDict({'NUM_BEV_FEATURES': 32})
            self.range_pillar_scatter = PointPillarScatter(pillar_scatter_cfg, self.grid_size)
            self.range_scatter = DynamicScatter(project_cfg['POINT_CLOUD_RANGE'], project_cfg['VOXEL_SIZE'], True)
            self.range_dynamic_voxelization = DynamicVoxelization(self.point_cloud_range, self.voxel_size, True)
        else:
            self.bev_projector_range = BEVProjector(project_cfg)

        self.with_pooling = self.model_cfg.get('WITH_POOLING', False)
        if self.with_pooling:
            self.pool = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        kernel_cfg = self.model_cfg.get('KERNEL_CFG', None)
        self.use_kernel = (kernel_cfg is not None)
        self.kernel_type = None
        if self.use_kernel:
            self.kernel_type = kernel_cfg.pop("type")
        if self.kernel_type == "meta":
            self.kernel = MetaKernel(**kernel_cfg)
        elif self.kernel_type == "edge_conv":
            self.kernel = EdgeConvKernel(**kernel_cfg)


    def forward(self,data_dict):
        if self.use_voxelization:
            return  self.forward_v2(data_dict)
        else:
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

        if self.fuse_complete_points:
            points_img_far = data_dict['points_img_far']
            proj_masks_far = data_dict['proj_masks_far']
            points_batch_far = points_img_far[:, :3].permute(0, 2, 3, 1).contiguous()  # B, H, W, 3
            points_batch_far = points_batch_far.view(batch_size, H * W, -1)  # B, H * W, 3
            proj_masks_far = proj_masks_far.view(batch_size, H * W)


        bev_features_batch = []
        for k in range(batch_size):
            points = points_batch[k]  # H * W, 4
            features = features_batch[k]  # H * W, C
            mask = proj_masks[k]  # H * W
            points = points[mask > 0, :]
            features = features[mask > 0, :]

            if self.fuse_complete_points:
                points_far=points_batch_far[k]
                mask_far=proj_masks_far[k]
                points_far=points_far[mask_far>0,:]

                # from tools.visual_utils.cloud_viewer import draw_lidar
                # draw_lidar(points.cpu().numpy()[:, :3],  'points_near')
                # draw_lidar(points_far.cpu().numpy()[:, :3], 'points_far')
                dist, idx = pointnet2_utils.three_nn(points_far.unsqueeze(0), points.unsqueeze(0))
                dist_recip = 1.0 / (dist + 1e-8)
                norm = torch.sum(dist_recip, dim=2, keepdim=True)
                weight = dist_recip / norm
                features_inter=features.unsqueeze(0).permute(0,2,1).contiguous()
                interpolated_feats = pointnet2_utils.three_interpolate(features_inter, idx, weight)
                interpolated_feats=interpolated_feats.permute(0,2,1).contiguous().squeeze(0) #N*C
                features=torch.cat([features,interpolated_feats],dim=0)
                points=torch.cat([points,points_far],dim=0)
                # draw_lidar(points.cpu().numpy()[:, :3], 'points_merge')
            # project points to BEV
            bev_features = self.bev_projector_range.get_projected_features(points, features, self.with_raw_features)  # C + 4, H2, W2
            bev_features_batch.append(bev_features)
        bev_features_batch = torch.stack(bev_features_batch,dim=0)  # B, C + 4, H2, W2

        # if self.fuse_complete_points:
        #     batch_complete_features = self.complete_net(batch_complete_points[:, 1:5])
        #     complete_feature_batch=[]
        #     for i in range(batch_size):
        #         mask_complete=batch_complete_points[:,0]==i
        #         points_complete=batch_complete_points[mask_complete]
        #         features_complete=batch_complete_features[mask_complete]
        #         complete_features=self.bev_projector_complete.get_projected_features(points_complete,features_complete,batch_size)
        #         complete_feature_batch.append(complete_features)
        #     complete_feature_batch=torch.stack(complete_feature_batch,dim=0)
        #     bev_features_batch=torch.cat([bev_features_batch,complete_feature_batch],dim=1)

        if self.with_pooling:
            bev_features_batch = self.pool(bev_features_batch)

        if self.use_kernel:
            assert self.with_raw_features
            mask = (bev_features_batch.sum(dim=1,) != 0).unsqueeze(1)
            x = torch.cat([bev_features_batch[:, :3],
                           bev_features_batch[:, 4:]], dim=1)
            bev_features_batch = self.kernel(x, mask)

        data_dict['spatial_features'] = bev_features_batch
        return data_dict


    def forward_v2(self, data_dict):
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
        batch_complete_points = data_dict['points']
        batch_size, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        features_batch = x.view(batch_size, H * W, C)  # B, H * W, C
        points_batch = points_img[:, :3].permute(0, 2, 3, 1).contiguous()  # B, H, W, 3
        points_batch = points_batch.view(batch_size, H * W, 3)  # B, H * W, 3
        proj_masks = proj_masks.view(batch_size,H*W)

        range_features_batch = []
        range_points_batch=[]
        for k in range(batch_size):
            points = points_batch[k]  # H * W, 4
            features = features_batch[k]  # H * W, C
            mask = proj_masks[k]  # H * W
            points = points[mask > 0, :]
            points = F.pad(points, (1, 0), mode='constant', value=k)
            range_points_batch.append(points)
            features = features[mask > 0, :]
            range_features_batch.append(features)
        range_points_batch=torch.cat(range_points_batch,dim=0)
        range_features_batch=torch.cat(range_features_batch,dim=0)
        # range_coord_batch=self.voxelize(range_points_batch)
        # voxel_feature_range,voxel_coord_range=self.range_scatter(range_features_batch,range_coord_batch)
        voxel_feature_range,voxel_coord_range=self.range_dynamic_voxelization(range_points_batch,range_features_batch)
        range_scatter_input={'pillar_features':voxel_feature_range,'voxel_coords':voxel_coord_range}
        bev_features_batch=self.range_pillar_scatter(range_scatter_input)['spatial_features']
        # print("type is ",type(batch_complete_points[:,1:5]))
        if self.fuse_complete_points:
            batch_complete_features = self.complete_net(batch_complete_points[:, 1:5])
            # batch_complete_coord=self.voxelize(batch_complete_points)
            # voxel_feature_complete, voxel_coord_complete = self.complete_scatter(batch_complete_features,
            #                                                                      batch_complete_coord)
            voxel_feature_complete, voxel_coord_complete = self.complete_dynamic_voxelization(batch_complete_points[:, 0:4],batch_complete_features)
            complete_scatter_input = {'pillar_features': voxel_feature_complete, 'voxel_coords': voxel_coord_complete}
            complete_feature_batch=self.complete_pillar_scatter(complete_scatter_input)['spatial_features']
            bev_features_batch=torch.cat([bev_features_batch,complete_feature_batch],dim=1)

        if self.with_pooling:
            bev_features_batch = self.pool(bev_features_batch)

        if self.use_kernel:
            assert self.with_raw_features
            mask = (bev_features_batch.sum(dim=1,) != 0).unsqueeze(1)
            x = torch.cat([bev_features_batch[:, :3],
                           bev_features_batch[:, 4:]], dim=1)
            bev_features_batch = self.kernel(x, mask)

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

