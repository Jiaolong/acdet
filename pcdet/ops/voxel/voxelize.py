# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
#from .voxel_layer import dynamic_voxelize, hard_voxelize
#from torch_scatter import scatter_max,scatter,scatter_mean

class _Voxelization(Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = hard_voxelize(points, voxels, coors,
                                      num_points_per_voxel, voxel_size,
                                      coors_range, max_points, max_voxels, 3)
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr



class DynamicVoxelization(nn.Module):
    def __init__(self,pc_range,voxel_size,average_points=True):
        super(DynamicVoxelization, self).__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        if not isinstance(pc_range,torch.Tensor):
            self.pc_range=torch.tensor(pc_range,dtype=torch.float32).cuda()
        if not isinstance(voxel_size,torch.Tensor):
            self.voxel_size=torch.tensor(voxel_size,dtype=torch.float32).cuda()
        self.pc_range=self.pc_range.reshape(1,-1)
        self.voxel_size=self.voxel_size.reshape(1,-1)
        self.average_points=average_points
        self.grid_size = (self.pc_range[:,3:6] - self.pc_range[:,0:3]) / self.voxel_size
        self.grid_size=self.grid_size[:,[2,1,0]]
        print("self grid size is ",self.grid_size)

    def forward(self,points ,features):
        # keep = (points[:, 0] >= self.pc_range[0,0]) & (points[:, 0] <= self.pc_range[0,3]) & \
        #     (points[:, 1] >= self.pc_range[0,1]) & (points[:, 1] <= self.pc_range[0,4]) & \
        #         (points[:, 2] >= self.pc_range[0,2]) & (points[:, 2] <= self.pc_range[0,5])
        # points = points[keep, :]
        # features=features[keep,:]
        batch_size=points[-1,0].item()+1
        voxels_batch=[]
        coors_batch=[]
        for i in range(int(batch_size)):
            cur_mask = points[:, 0] == i
            cur_points = points[cur_mask][:, 1:].contiguous()
            cur_features=features[cur_mask].contiguous()
            coords = ((cur_points[:, [2, 1, 0]] - self.pc_range[:, [2, 1, 0]]) / self.voxel_size[:, [2, 1, 0]]).to(torch.int64)
            keep = ((coords < self.grid_size) & (coords >= 0)).all(dim=1)
            cur_features=cur_features[keep,:]
            coords=coords[keep,:]
            unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
            if self.average_points:
                voxels = scatter_mean(cur_features, inverse_indices, dim=0)
            else:
                voxels=scatter_max(cur_features, inverse_indices, dim=0)[0]
            voxels_batch.append(voxels)
            unique_coords=F.pad(unique_coords, (1, 0), mode='constant', value=i)
            coors_batch.append(unique_coords)
        voxels_batch=torch.cat(voxels_batch,dim=0)
        coors_batch=torch.cat(coors_batch,dim=0)
        return voxels_batch,coors_batch


