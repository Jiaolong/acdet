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
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    result_kitti_path_gt = "/home/zhangxiao/code/pillars/OpenPCDet/data/kitti/kitti_infos_val.pkl"
    # result_kitti_path = "/home/zhangxiao/code/pillars/OpenPCDet/output/kitti_models/cyv_det_fuse2_raw_transformer_mask/default/eval/epoch_79/val/default/result.pkl"
    result_kitti_path = "/home/zhangxiao/tmp/pkl/range_det_meta_att_reduced_yolox_result.pkl"
    # result_kitti_path = "/home/zhangxiao/tmp/pkl/bev_result.pkl"

    if os.path.exists(result_kitti_path_gt):
        with open(result_kitti_path_gt, 'rb') as f:
            result_kitti_gt = pickle.load(f)
            # result_kitti_gt = sorted(result_kitti_gt, key=lambda i:int(i["point_cloud"]["lidar_idx"]))

    else:
        result_kitti_gt = None

    with open(result_kitti_path, 'rb') as f:
        result_kitti = pickle.load(f)
    cls_id_dict = {"Car":1, "Pedestrian":2, "Cyclist":3}

    idx_list = [614]
    with torch.no_grad():
        for idx in range(len(result_kitti)):
            frame_id = int(result_kitti[idx]["frame_id"])
            if frame_id not in idx_list:
                continue
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset[frame_id]
            data_dict = demo_dataset.collate_batch([data_dict])
            # load_data_to_gpu(data_dict)
            # pred_dicts, _ = model.forward(data_dict)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            sample_result = result_kitti[idx]
            # keep_index = [name in ["Car", "Pedestrian", "Cyclist"] for name in sample_result["name"]]
            keep_index = sample_result["score"] > 0.4

            bbox = sample_result["boxes_lidar"]
            bbox = bbox[keep_index[:len(bbox)]]
            labels = sample_result["name"][keep_index]
            labels = torch.from_numpy(np.array([cls_id_dict[cls_id] for cls_id in labels]))

            if result_kitti_gt is not None:
                samplt_result_gt = result_kitti_gt[idx]['annos']
                keep_index = [name in ["Car", "Pedestrian", "Cyclist"] for name in samplt_result_gt["name"]]
                bbox_gt = samplt_result_gt["gt_boxes_lidar"]
                bbox_gt = bbox_gt[keep_index[:len(bbox_gt)]]
            
            if result_kitti_gt is not None:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:], gt_boxes=bbox_gt,
                    ref_boxes=bbox, ref_labels=None,
                )
            else:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=bbox, ref_labels=labels
                )    
            
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
