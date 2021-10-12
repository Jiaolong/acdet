# import mayavi.mlab as mlab
#
import numpy as np
# def draw_lidar(pc, fig=None, color=None, scale=1., axis=True, show=False):
#     import mayavi.mlab as mlab
#     if fig is None: fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
#     ''' Draw lidar points. simplest set up. '''
#     if color is None: color = pc[:, 2]
#     # draw points
#     mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=scale,
#                   figure=fig)
#     if axis:
#         # draw origin
#         mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
#         # draw axis
#         axes = np.array([
#             [2., 0., 0., 0.],
#             [0., 2., 0., 0.],
#             [0., 0., 2., 0.],
#         ], dtype=np.float64)
#         mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
#         mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
#         mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
#     mlab.view(azimuth=180, elevation=70, focalpoint=[15, 0, 0], distance=50.0, figure=fig)
#     if show:
#         mlab.show()
#     return fig

import open3d as o3d

def draw_lidar(cloud,name="Open3D",colors=None):
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(cloud)
    if colors is not None:
        pcd.colors=o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([1,1,1])
    # o3d.visualization.draw_geometries([pcd],window_name=
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size=2
    opt.show_coordinate_frame=True
    opt.background_color = np.asarray([0, 0, 0])

    # opt.point_color_option.Color=np.asarray([1,1,1])
    vis.run()
    vis.destroy_window()