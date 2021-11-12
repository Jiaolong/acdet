import copy
import pickle
import os

import numpy as np
from skimage import io
import torch

from pcdet.utils import calibration_kitti, object3d_kitti, box_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import nms_weighted_gpu


path_pp = "/home/wangguojun/source_code/OpenPCDet/output/cfgs/kitti_models/cyv_det_metav6_fuse2_raw_transformer_mask/final_car_ped_cyc"
path_se = "/home/wangguojun/source_code/OpenPCDet/output/cfgs/kitti_models/cyv_det_metav6_fuse2_raw_transformer_mask_test_cyclist_ped/label_merge"
save_path = "/home/wangguojun/source_code/OpenPCDet/output/cfgs/kitti_models/cyv_det_metav6_fuse2_raw_transformer_mask_test_cyclist_ped/multi_model_label"
calib_root_path = "/data/kitti/testing/calib"
from pathlib import Path
Path(save_path).mkdir(exist_ok=True,parents=True)
def get_label_from_path(label_file):
    sample_idx = label_file.split('/')[-1].split('.')[0]
    calib_file_path = os.path.join(calib_root_path, sample_idx + ".txt")
    assert os.path.exists(label_file)
    obj_list = object3d_kitti.get_objects_from_label(label_file)

    assert os.path.exists(calib_file_path)
    calib = calibration_kitti.Calibration(calib_file_path)
    # print("leng obj is ",len(obj_list))
    if len(obj_list)>0:
        annotations = {}
        annotations['name'] = np.array([obj.cls_type for obj in obj_list])
        annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
        annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
        annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
        annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
        annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
        annotations['score'] = np.array([obj.score for obj in obj_list])
        annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
        num_gt = len(annotations['name'])
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)

        loc = annotations['location'][:num_objects]
        dims = annotations['dimensions'][:num_objects]
        rots = annotations['rotation_y'][:num_objects]
        loc_lidar = calib.rect_to_lidar(loc)
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
        loc_lidar[:, 2] += h[:, 0] / 2

        score = annotations['score'][:num_objects]
        name = annotations['name'][:num_objects]
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis]),
                                         score[..., np.newaxis], name[..., np.newaxis]], axis=1)
    else:
        gt_boxes_lidar=np.zeros((0,9))

    return gt_boxes_lidar, calib

def get_template_prediction(num_samples):
    ret_dict = {
        'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
        'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
        'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
        'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
        'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
    }
    return ret_dict

def generate_single_sample_dict(box_dict, calib):
    pred_scores = box_dict['pred_scores'].cpu().numpy()
    pred_boxes = box_dict['pred_boxes'].cpu().numpy()
    pred_labels = box_dict['pred_labels'].cpu().numpy()
    pred_dict = get_template_prediction(pred_scores.shape[0])
    if pred_scores.shape[0] == 0:
        return pred_dict

    pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
    pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        pred_boxes_camera, calib, image_shape=(375, 1242)
    )

    pred_dict['name'] = np.array(cls_list)[pred_labels]
    pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
    pred_dict['bbox'] = pred_boxes_img
    pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
    pred_dict['location'] = pred_boxes_camera[:, 0:3]
    pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
    pred_dict['score'] = pred_scores
    pred_dict['boxes_lidar'] = pred_boxes

    return pred_dict


all_files = os.listdir(path_pp)
all_files = sorted(all_files, key = lambda x : int(x.split('.')[0]))

cls_list = ["Car", "Pedestrian", "Cyclist"]

"""
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
"""

for ann_file_name in all_files:
    file_path_m = os.path.join(path_pp, ann_file_name)
    file_path_u = os.path.join(path_se, ann_file_name)


    gt_bbox_lidar_m, _ = get_label_from_path(file_path_m)
    gt_bbox_lidar_u, calib = get_label_from_path(file_path_u)
    gt_bbox = np.concatenate((gt_bbox_lidar_m, gt_bbox_lidar_u), axis=0)
    gt_bbox = sorted(gt_bbox, key=lambda x : x[7], reverse=True)
    names = np.array([box[-1] for box in gt_bbox])
    bboxes = [box[:-1] for box in gt_bbox]
    bboxes = torch.from_numpy(np.array(bboxes, np.float64)).cuda().float()
    

    if bboxes.shape[0]>0:
        box_dict = {}
        for idx, cls_ in enumerate(cls_list):
            cls_index = (names == cls_)
            bbox = bboxes[cls_index]
            score = bbox[:, -1]
            if bbox.size(0) != 0:
                bbox, keep = nms_weighted_gpu(bbox[:, :7].contiguous(), score, thresh=0.1)
                if "pred_scores" not in box_dict.keys():
                    box_dict["pred_scores"] = score[keep]
                    box_dict["pred_boxes"] = bbox[keep]
                    box_dict["pred_labels"] = torch.zeros_like(score[keep]).int() + idx

                else:
                    box_dict["pred_scores"] = torch.cat((box_dict["pred_scores"], score[keep]))
                    box_dict["pred_boxes"] = torch.cat((box_dict["pred_boxes"], bbox[keep]), dim=0)
                    tmp_score = torch.zeros_like(score[keep]).int() + idx
                    box_dict["pred_labels"] = torch.cat((box_dict["pred_labels"], tmp_score))
    else:
        box_dict = {
            'pred_scores': torch.zeros(0).cuda(),
            'pred_boxes': torch.zeros([0, 7]).cuda(),
            'pred_labels': torch.zeros(0).cuda(),
        }

    single_pred_dict = generate_single_sample_dict(box_dict, calib)

    cur_det_file = os.path.join(save_path, ann_file_name)
    with open(cur_det_file, 'w') as f:
        bbox = single_pred_dict['bbox']
        loc = single_pred_dict['location']
        dims = single_pred_dict['dimensions']  # lhw -> hwl

        for idx in range(len(bbox)):
            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                    % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                        bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                        dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                        loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                        single_pred_dict['score'][idx]), file=f)
