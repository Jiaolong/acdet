import torch
from torch import nn
from torch.nn import functional as F

import numba
import numpy as np
import cv2
from .vis_utils import center_to_corner_box2d, kitti_vis
from pcdet.utils.box_utils import boxes3d_lidar_to_aligned_bev_boxes, boxes_to_corners_3d

def gaussian_2d(shape, sigma=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

label_radius = {1:6, 2:6, 3:6}
radius_gaussion = {6:gaussian_2d((13,13), 6/6)}

class GeneragePillarMask(object):
    """
    """
    def __init__(self,
                 voxel_size=(0.16, 0.16, 4),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1),):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.feature_h = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1] + 0.5)
        self.feature_w = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0] + 0.5)

    def generate_mask(self, points, vis_voxel_size, vis_point_range, boxes, scale=2, use_dym_ratio=False):
        """generate segmask by given pointcloud and bounding boxes

        Args:
            points (torch.Tensor): point cloud batch
            vis_voxel_size (list): voxel size
            vis_point_range (list): point cloud range
            boxes (LiDARInstance3DBoxes): gt boxes
        """
        if boxes is None:
            return None
        w = int((vis_point_range[4] - vis_point_range[1]) / vis_voxel_size[1] + 0.5)
        h = int((vis_point_range[3] - vis_point_range[0]) / vis_voxel_size[0] + 0.5)
        batch_size = points[-1][0].int().item() + 1
        segmask_maps = np.zeros((batch_size, int(w/scale), int(h/scale)))
        for i in range(segmask_maps.shape[0]):
            vis_point_range = np.array(vis_point_range)
            if isinstance(boxes[i], list):
                assert len(boxes[i]) == 1
                current_bbox = boxes[i][0].detach().cpu().numpy()
            else:
                current_bbox = boxes[i].detach().cpu().numpy()
            bev_corners = center_to_corner_box2d(
                current_bbox[:, [0, 1]], current_bbox[:, [3, 4]], current_bbox[:, 6])
            bev_corners -= vis_point_range[:2]
            bev_corners *= np.array(
                (w, h))[::-1] / (vis_point_range[3:5] - vis_point_range[:2])
            bev_corners = bev_corners / scale
            segmask = np.zeros((w//scale, h//scale, 3))
            if use_dym_ratio:
                for idx in range(current_bbox.shape[0]):
                    radius = max(0.5 * 1.5 * min(current_bbox[idx, [3, 4]] * np.array((w, h))[::-1] / (vis_point_range[3:5] - vis_point_range[:2])),
                                6.25)
                    segmask = cv2.drawContours(segmask, [bev_corners[idx].astype(np.int32)],
                                            -1, int(radius+0.5), -1)
            else:
                for idx in np.unique(current_bbox[:,-1]):
                    segmask = cv2.drawContours(segmask, bev_corners[current_bbox[:, -1] == idx].astype(np.int),
                                            -1, int(idx), -1)

            segmask_maps[i] = segmask[:, :, 0]
        return segmask_maps

    def generate(self, masks, points, coors, vis_voxel_size, vis_point_range, gt_bboxes_3d):
        use_dym_ratio = True
        scale = self.feature_h // masks.size(2)
        segmask_maps = self.generate_mask(points, vis_voxel_size=vis_voxel_size,
                                vis_point_range=vis_point_range,
                                boxes=gt_bboxes_3d, scale=scale, use_dym_ratio=use_dym_ratio)
        heatmap = generate_gaussion_heatmap_array(np.array(masks.size()),
                                                coors.cpu().numpy(),
                                                segmask_maps, scale, use_dym_ratio=use_dym_ratio)
        heatmap = torch.from_numpy(heatmap)
        return heatmap

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)

def generate_gaussion_heatmap_array(heatmap_size, coors, segmask_maps, scale=2, use_dym_ratio=False):
    """generate gaussiin heatmap

    Args:
        heatmap_size (numpy array): [b, c , w, h]
        coors (list): pillar coors
        segmask_maps (np.ndarray): [description]
        gaussian (np.ndarray): gaussion heatmap generate by radius
        scale (int, optional): [description]. Defaults to 2.

    Returns:
        [np.ndarray]: gaussion heatmap
    """
    heatmap = np.zeros((heatmap_size[0], heatmap_size[2], heatmap_size[3]))
    coors_int = coors.astype(np.int64)
    for i in range(coors_int.shape[0]):
        batch_idx = coors_int[i][0]
        center = coors_int[i][-2:][::-1] // scale
        label_idx = segmask_maps[batch_idx, center[1], center[0]]
        if  label_idx == 0:
            continue
        
        if use_dym_ratio:
            radius = label_idx
            diameter = 2 * radius + 1
            gaussian = gaussian_2d((diameter, diameter), sigma=diameter/6)
        else:
            radius = label_radius[label_idx]
            gaussian = radius_gaussion[radius]
        draw_heatmap_gaussian_array(heatmap[batch_idx], center, radius, gaussian)
    return heatmap

@numba.jit(nopython=True)
def draw_heatmap_gaussian_array(heatmap, center, radius, gaussian, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (np.ndarray): Heatmap to be masked.
        center (np.ndarray): Center coord of the heatmap.
        radius (int): Radius of gausian.
        K (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        np.ndarray: Masked heatmap.
    """

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                                radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        masked_heatmap = np.maximum(masked_heatmap, masked_gaussian * k)
        heatmap[y - top:y + bottom, x - left:x + right] = masked_heatmap
    return heatmap