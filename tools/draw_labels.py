import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from visual_utils.visualize_utils_v2 import show_lidar_with_boxes

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_dir', type=str, default='/data/kitti/training/velodyne',
                        help='specify the config for demo')
    parser.add_argument('--label_dir', type=str, default='/data/kitti/training/label_2',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--calib_dir',type=str,default='/data/kitti/training/calib')

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    points_files=glob.glob(str(Path(args.data_dir).resolve())+'/*.bin')
    points_files.sort()
    label_files=glob.glob(str(Path(args.label_dir).resolve())+'/*.txt')
    label_files.sort()
    class_names=['Car', 'Pedestrian', 'Cyclist','Van','Person_sitting','Truck','DontCare','Misc']
    for label_file in label_files:
        frame_id=Path(label_file).stem
        print("current frame is ",frame_id)
        points_file=Path(args.data_dir)/(frame_id+".bin")
        calib_file=Path(args.calib_dir)/(frame_id+".txt")
        points=np.fromfile(str(points_file), dtype=np.float32).reshape(-1, 4)
        objects_cam=object3d_kitti.get_objects_from_label(label_file)
        gt_boxes_camera=np.zeros((len(objects_cam),8))
        calib=calibration_kitti.Calibration(calib_file)
        if len(objects_cam)>0:
            gt_boxes_camera[:,3:6] = np.array([[obj.l, obj.h, obj.w] for obj in objects_cam])  # lhw(camera) format
            gt_boxes_camera[:,:3] = np.concatenate([obj.loc.reshape(1, 3) for obj in objects_cam], axis=0)
            gt_boxes_camera[:,6] = np.array([obj.ry for obj in objects_cam])
            gt_boxes_camera[:,7]=np.array([obj.score for obj in objects_cam ])
            gt_names=np.array([obj.cls_type for obj in objects_cam])
            gt_classes = np.array([class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib).astype(np.float32)
            V.draw_scenes(points=points,ref_scores=gt_boxes_camera[:,7], ref_boxes=gt_boxes_lidar,ref_labels=gt_classes)
            # show_lidar_with_boxes(points=points,boxes3d=gt_boxes_lidar)
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
