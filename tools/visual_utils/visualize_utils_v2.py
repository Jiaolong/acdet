import open3d as o3d
import numpy as np
from pcdet.utils.box_utils import boxes_to_corners_3d
import time
lines = [
    [0, 1],
    [1, 2],
    [0, 3],
    [2, 3],
    [4, 5],
    [5, 6],
    [4, 7],
    [6, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7]]

color_list = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 1]
]


def show_lidar_list(points_list):
    geo_objs = []
    for i, points in enumerate(points_list):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points[:, :3])
        colors_pc = [color_list[i] for j in range(points.shape[0])]
        pc.colors = o3d.utility.Vector3dVector(colors_pc)
        geo_objs.append(pc)

    o3d.visualization.draw_geometries(geo_objs)


def show_lidar_with_boxes(points, boxes3d=None, labels=None, is_corners=False):
    if points is None:
        return

    geo_objs = boxes2lineset(boxes3d, labels, is_corners)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    pc.paint_uniform_color([0.5, 0.5, 0.5])
    geo_objs.append(pc)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    geo_objs.append(axis_pcd)

    o3d.visualization.draw_geometries(geo_objs)


def boxes2lineset(boxes3d, labels=None, is_corners=False):
    geo_objs = []
    if boxes3d is not None:
        if not is_corners:
            corners_arr = boxes_to_corners_3d(boxes3d)
        else:
            corners_arr = boxes3d

        for i, corners in enumerate(corners_arr):
            color = [1, 0, 0]
            if labels is not None:
                color = color_list[labels[i] - 1]
            colors = [color for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(corners),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geo_objs.append(line_set)
    return geo_objs


class LidarDetPlot():
    """
    Plot 3D detections and point cloud in dynamic frames
    ref: https://github.com/Jiang-Muyun/Open3D-Semantic-KITTI-Vis/blob/ddb188e1375a1d464dec077826725afd72a85e63/src/kitti_base.py#L43
    """

    def __init__(self, name="LidarDetPlot", width=800, height=600):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=name, width=width, height=height, left=100)
        self.axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        opt.point_size = 1
        opt.show_coordinate_frame = True

        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.vis.register_key_callback(32, lambda vis: exit())

    def __del__(self):
        self.vis.destroy_window()

    def update(self, points, boxes=None, labels=None, sleep=1):

        if points is None:
            return

        self.pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        self.pcd.paint_uniform_color([1.0, 1.0, 1.0])

        self.vis.clear_geometries()
        # self.vis.update_geometry(self.pcd)
        # self.vis.remove_geometry(self.pcd)
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.axis_pcd)

        geo_objs = boxes2lineset(boxes, labels)
        for geo_obj in geo_objs:
            self.vis.add_geometry(geo_obj)

        self.vis.poll_events()
        self.vis.update_renderer()

        if sleep > 0:
            time.sleep(sleep)