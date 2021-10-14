import numpy as np
import torch

from torch import nn as nn


class EncoderBlock(nn.Module):
    """
    """
    def __init__(self,
                 in_channels=128,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(EncoderBlock, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for idx, layer_num in enumerate(layer_nums):
            block = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    in_filters[idx], out_channels[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(out_channels[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for j in range(layer_num):
                block.append(nn.Conv2d(out_channels[idx], out_channels[idx],
                                       kernel_size=3, padding=1, bias=False))
                block.append(nn.BatchNorm2d(out_channels[idx], eps=1e-3, momentum=0.01),)
                block.append(nn.ReLU())

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)

class FeatuteExtract(nn.Module):
    """
    """
    def __init__(self,
                 in_channels=4,
                 with_distance=False,
                 with_cluster_center=False,
                 voxel_size=(0.16, 0.16, 4),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),
                 num_scale_feature=None):
        super(FeatuteExtract, self).__init__()
        self.in_channels = in_channels
        self._with_cluster_center = with_cluster_center
        self.min_height = point_cloud_range[2]
        self.feature_h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1] + 0.5)
        self.feature_w = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0] + 0.5)
        self.num_scale_feature = num_scale_feature

        if self.num_scale_feature is not None:
            assert len(self.num_scale_feature) > 0
            self.num_scale_feature = [in_channels] + self.num_scale_feature
            self.scale_layers = nn.ModuleList()
            
            for i in range(len(self.num_scale_feature) - 1):
                in_cells = self.num_scale_feature[i]
                out_cells = self.num_scale_feature[i+1]
                self.scale_layers.append(nn.Sequential(
                    nn.Linear(in_cells, out_cells, bias=False),
                    nn.BatchNorm1d(out_cells, eps=1e-3, momentum=0.01),
                    nn.ReLU())
                )

    def forward(self, batch_dict, **kwargs):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """

        features, num_points, coors = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        batch_size = coors[:, 0].max().int().item() + 1
        features_ls = [num_points.unsqueeze(1)]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)
        if self._with_cluster_center:
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Get max height point at pillars
        # voxel_num = features.size(1)
        # mask = get_paddings_indicator(num_points, voxel_num)
        # features[:, :, 2][~mask] = self.min_height
        # voxel_height_max, voxel_height_max_idx = features[:, :, 2].max(dim=1, keepdim=True)
        # features[:, :, 2][~mask] = 0
        # features_ls.append(voxel_height_max)

        # Get mean height 
        voxel_height_mean = features[:, :, 2].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1)
        # features_ls.append(voxel_height_mean)

        # Get top intensity
        # top_intensity = features[:, :, -1].gather(1, voxel_height_max_idx)
        # mean_intensity = features[:, :, -1].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1)
        # features_ls.append(top_intensity)
        # features_ls.append(mean_intensity)

        # Distance
        points_dist = torch.norm(features[:, :, :3], 2, 2)
        points_dist = points_dist.sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1)
        d_point_mean = torch.norm(points_mean, 2, 2, keepdim=True)
        features_ls.append(points_mean.squeeze(1))
        features_ls.append(d_point_mean.squeeze(1))

        voxel_features = torch.cat(features_ls, dim=-1)
        if self.num_scale_feature is not None:
            for i in range(len(self.scale_layers)):
                voxel_features = self.scale_layers[i](voxel_features)
        self.in_channels = voxel_features.size(1)

        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.in_channels, self.feature_h * self.feature_w,
                                 dtype=voxel_features.dtype, device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt

            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.feature_w + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.feature_h,
                                         self.feature_w)
        # cv2.imwrite("/home/zhangxiao/test_1.png", (batch_canvas[0][0] * 255).cpu().numpy().astype(np.uint8))
        return batch_canvas

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(self.in_channels, self.feature_h * self.feature_w,
                                dtype=voxel_features.dtype, device=voxel_features.device)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.feature_h, self.feature_w)
        return [canvas]


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Use {} device".format(device))

    x = torch.randn((16, 6, 468, 468)).float().to(device)
    with torch.no_grad():
        scaleBlock = ScaleAwareBlock(in_channels=6, layer_nums=[2,3,3], layer_strides=[1,2,2]).to(device)
    out = scaleBlock(x)
    for i in out:
        print("Size of output is {}".format(i.size()))