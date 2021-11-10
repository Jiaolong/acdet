import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import pickle
import os

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


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
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # result_kitti_path_gt = "/home/zhangxiao/code/pillars/OpenPCDet/data/kitti/kitti_infos_trainval.pkl"
    result_waymo_path_gt = "/data/waymo/pcdet_data/waymo/waymo_infos_val.pkl"
    # result_kitti_path = "/home/zhangxiao/tmp/pkl/range_det_meta_att_reduced_yolox_result.pkl"
    result_waymo_path = "/home/zhangxiao/code/pillars/OpenPCDet/output/waymo_models/cyv_det_metav4/default/eval/epoch_18/val/default/result.pkl"
    root_path = "/data/waymo/pcdet_data/waymo/waymo_processed_data"

    if os.path.exists(result_waymo_path_gt):
        with open(result_waymo_path_gt, 'rb') as f:
            result_waymo_gt = pickle.load(f)
    else:
        result_waymo_gt = None

    with open(result_waymo_path, 'rb') as f:
        result_waymo = pickle.load(f)
    # result_kitti = sorted(result_kitti, key=lambda i:int(i["point_cloud"]["lidar_idx"]))
    cls_id_dict = {"Vehicle":1, "Pedestrian":2, "Cyclist":3}

    idx_list = [2232]
    with torch.no_grad():
        for idx in range(len(result_waymo)):
            if idx < 123:
                continue
            point_cloud = result_waymo_gt[idx * 5]["point_cloud"]
            lidar_sequence = point_cloud["lidar_sequence"]
            sample_idx = point_cloud["sample_idx"]
            point_cloud_path = os.path.join(root_path, lidar_sequence, str(sample_idx).zfill(4)+".npy")
            points = np.load(point_cloud_path)

            logger.info(f'Visualized sample index: \t{idx + 1}')

            samplt_result_gt = result_waymo_gt[idx * 5]["annos"]
            keep_index_gt = [name in ["Vehicle", "Pedestrian", "Cyclist"] for name in samplt_result_gt["name"]]
            filer_by_num_points = samplt_result_gt["num_points_in_gt"] > 5
            bbox_gt = samplt_result_gt["gt_boxes_lidar"][(keep_index_gt&filer_by_num_points)]

            if result_waymo is not None:
                samplt_result = result_waymo[idx]
                keep_idx = samplt_result["score"] > 0.5
                bbox = samplt_result['boxes_lidar'][keep_idx]
                score = samplt_result["score"][keep_idx]
                V.draw_scenes(
                    points=points, gt_boxes=bbox_gt, ref_boxes=bbox, ref_labels=None, ref_scores=score
                )

            else:
                V.draw_scenes(
                    points=points, gt_boxes=bbox_gt,
                    ref_boxes=None, ref_labels=None
                )
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
