import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from pcdet.datasets import KittiDataset
from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu
import tqdm
from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # np.random.seed(1024)
    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    logger = common_utils.create_logger()

    num_sample = 1000
    H = 48
    W = 512
    r = 0.5
    # H_bev= 496
    # W_bev=432
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    voxel_size = [0.16, 0.16, 4]

    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_num_points=50,
        max_voxels=18000
    )
    cfg.DATA_CONFIG.POINT_CLOUD_RANGE=point_cloud_range

    for cur_cfg in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if cur_cfg.NAME=='transform_points_to_voxels':
            cur_cfg.VOXEL_SIZE=voxel_size

    for cur_cfg in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if cur_cfg.NAME=="project_points":
            cur_cfg.NUM_COLS=W
            cur_cfg.NUM_ROWS=H
    if cfg.MODEL.get('BACKBONE_FV',None):
        if cfg.MODEL.BACKBONE_FV.get('KERNEL_CFG',None):
            cfg.MODEL.BACKBONE_FV.KERNEL_CFG.FEATURE_MAP_SIZE=[H,W]

    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array([0.16, 0.16, 4])
    grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    H_bev= grid_size[1]
    W_bev=grid_size[0]

    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)



    with torch.no_grad():
        # load checkpoint
        if args.ckpt is not None:
            model.load_params_from_file(filename=args.ckpt, logger=logger)
        model.cuda()

        logger.info('*************** SPEED EVALUATION *****************')

        model.eval()

        batch_dict = {}
        batch_dict['points_img'] = torch.rand((1, 5, H, W), dtype=torch.float).cuda()
        batch_dict['points_img_far'] = torch.rand((1, 5, H, W), dtype=torch.float).cuda()
        batch_dict['batch_size'] = 1
        mask = torch.ones((1, H, W)).cuda().bool()
        mask[:, :int(H * r), :] = False
        batch_dict['proj_masks'] = mask
        mask_far = torch.ones((1, H, W)).cuda().bool()
        mask_far[:, :int(H * r), :] = False
        batch_dict['proj_masks_far'] = mask_far

        batch_dict['points_img_bev'] =torch.rand((1, 3, H_bev, W_bev), dtype=torch.float).cuda()
        mask_bev = torch.ones((1, H_bev, W_bev)).cuda().bool()
        mask_bev[:, :int(H_bev * r), :] = False
        batch_dict['proj_masks_bev'] =mask_bev
        cloud_range=np.zeros(4,dtype=np.float)
        cloud_min = np.zeros(4, dtype=np.float)
        cloud_range[:3]=point_cloud_range[3:]-point_cloud_range[:3]
        cloud_min[:3]=point_cloud_range[:3]
        cloud_range[3]=100
        cloud_range=cloud_range.reshape(1,4)
        cloud_min= cloud_min.reshape(1,4)

        batch_dict['points']=np.random.rand(24000,4)*cloud_range+cloud_min
        voxel_output = voxel_generator.generate(batch_dict['points'])
        batch_dict['voxels']=torch.from_numpy(voxel_output['voxels']).cuda().float()
        coordinates = np.pad(voxel_output['coordinates'], ((0, 0), (1, 0)), mode='constant', constant_values=0)
        batch_dict['voxel_coords']=torch.from_numpy(coordinates).cuda().long()
        batch_dict['voxel_num_points']=torch.from_numpy(voxel_output['num_points_per_voxel']).cuda()

        progress_bar = tqdm.tqdm(total=num_sample, leave=True, desc='eval', dynamic_ncols=True)
        start_time = time.time()
        for i in range(num_sample):
            with torch.no_grad():
                _, _ = model(batch_dict)
            disp_dict = {}
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
        progress_bar.close()
        logger.info('*************** Speed Performance  %s *****************')
        sec_per_example = (time.time() - start_time) / num_sample
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)
        logger.info('****************Evaluation done.*****************')


if __name__ == '__main__':
    main()
