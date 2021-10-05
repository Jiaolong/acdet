import numpy as np
import torch
import torch.nn.functional as F

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils
from ....utils import common_utils


class SimOTAMultiClassTargetAssigner(object):
    """
    """
    def __init__(self,
                 box_coder,
                 model_cfg,
                 class_names,
                 topk=10,
                 center_radius=0.3,
                 cls_weight=1.0,
                 iou_weight=3.0,
                 match_height=False):
        self.box_coder = box_coder
        self.topk = topk
        self.center_radius = center_radius
        self.cls_weight = cls_weight
        self.iou_weight = iou_weight
        self.match_height = match_height

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]

    def assign_targets(self,
                       anchors_list,
                       gt_boxes_with_classes,
                       batch_cls_preds=None,
                       batch_box_preds=None):
        """
        Args:
            anchors_list: [(N, 7), ...]
            gt_boxes_with_classes: (B, M, 8)
            batch_cls_preds: (B, N, num_class)
            batch_box_preds: (B, N, 7)
        Returns:

        """
        bbox_targets = []
        iou_targets = []
        cls_labels = []
        reg_weights = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]

        batch_cls_labels, batch_reg_targets, batch_iou_targets, batch_reg_weights = [], [], [], []
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()
            
            cur_cls_preds = batch_cls_preds[k]
            cur_box_preds = batch_box_preds[k]

            cls_labels, reg_targets, iou_targets, reg_weights = [], [], [], []
            i = 0
            for anchor_class_name, anchors in zip(self.anchor_class_names, anchors_list):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                feature_map_size = anchors.shape[:3]
                anchors = anchors.view(-1, anchors.shape[-1])
                selected_classes = cur_gt_classes[mask]
                
                num_anchors = anchors.shape[0]
                cur_cls_labels, cur_reg_targets, cur_iou_targets, cur_reg_weights = self.assign_targets_single(
                        anchors, cur_gt[mask], selected_classes, 
                        cur_cls_preds[i * num_anchors: (i + 1) * num_anchors], 
                        cur_box_preds[i * num_anchors: (i + 1) * num_anchors])
                
                i += 1
                cls_labels.append(cur_cls_labels)
                reg_targets.append(cur_reg_targets)
                iou_targets.append(cur_iou_targets)
                reg_weights.append(cur_reg_weights)
            
            cls_labels = torch.cat(cls_labels, dim=0)
            reg_targets = torch.cat(reg_targets, dim=0)
            iou_targets = torch.cat(iou_targets, dim=0)
            reg_weights = torch.cat(reg_weights, dim=0)

            batch_cls_labels.append(cls_labels)
            batch_reg_targets.append(reg_targets)
            batch_iou_targets.append(iou_targets)
            batch_reg_weights.append(reg_weights)
        
        
        ret_dict = {
            'box_cls_labels': torch.stack(batch_cls_labels, dim=0),
            'box_reg_targets': torch.stack(batch_reg_targets, dim=0),
            'box_iou_targets': torch.stack(batch_iou_targets, dim=0),
            'reg_weights': torch.stack(batch_reg_weights, dim=0)
        }

        return ret_dict

    def assign_targets_single(self,
                              anchors,
                              gt_boxes,
                              gt_classes,
                              cls_preds,
                              box_preds,
                              eps=1e-7):
        """
        Args:
            anchors: (N, 7) [x, y, z, dx, dy, dz, heading]
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
            gt_classes: (M)
            cls_preds: (N, num_class)
            box_preds: (N, 7)
        Returns:

        """
        INF = 100000000
        num_anchor = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        num_boxes = box_preds.shape[0]
        assert num_anchor == num_boxes
        
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            anchors, gt_boxes)

        valid_boxes = box_preds[valid_mask]
        valid_scores = cls_preds[valid_mask]
        num_valid = valid_boxes.size(0)

        # compute pairwise ious
        if self.match_height:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(valid_boxes[:, 0:7],
                                                   gt_boxes[:, 0:7])  # (N, M)
        else:
            ious = iou3d_nms_utils.boxes_iou_bev(valid_boxes[:, 0:7],
                                                 gt_boxes[:, 0:7])
            #ious = box_utils.boxes3d_nearest_bev_iou(valid_boxes[:, 0:7],
            #                                     gt_boxes[:, 0:7])

        iou_cost = -torch.log(ious + eps)

        # gt_classes is 1-based index
        gt_onehot_label = F.one_hot((gt_classes - 1).to(torch.int64), cls_preds.shape[-1]).float()
        gt_onehot_label = gt_onehot_label.unsqueeze(0).repeat(num_valid, 1, 1)
        valid_scores = valid_scores.unsqueeze(1).repeat(1, num_gt, 1)

        cls_cost = F.binary_cross_entropy(valid_scores.sqrt_(),
                                          gt_onehot_label,
                                          reduction='none').sum(-1)

        cost_matrix = (cls_cost * self.cls_weight +
                       iou_cost * self.iou_weight +
                       (~is_in_boxes_and_center) * INF)

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, ious, num_gt, valid_mask)

        matched_gts = box_preds.new_full((num_anchor, self.box_coder.code_size), 0)
        matched_gts[valid_mask] = gt_boxes[matched_gt_inds]
        cls_labels = matched_gt_inds.new_full((num_anchor, ), -1)
        cls_labels[valid_mask] = gt_classes[matched_gt_inds].long()
        iou_targets = matched_gt_inds.new_full((num_anchor, ), 0.0, dtype=torch.float32)
        iou_targets[valid_mask] = matched_pred_ious

        reg_targets = matched_gts.new_zeros(
            (num_anchor, self.box_coder.code_size))
        pos_mask = cls_labels > 0
        reg_weights = matched_gts.new_zeros(num_anchor)

        if pos_mask.sum() > 0:
            reg_targets[pos_mask > 0] = self.box_coder.encode_torch(
                matched_gts[pos_mask > 0], anchors[pos_mask > 0])
            reg_weights[pos_mask] = 1.0
        
        return cls_labels, reg_targets, iou_targets, reg_weights

    def get_in_gt_and_in_center_info(self, anchors, gt_boxes):
        """
        Args:
            anchors: (N, 7) [x, y, z, dx, dy, dz, heading]
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
        """
        MARGIN = 1e-2
        num_gt = gt_boxes.size(0)

        repeated_x = anchors[:, 0].unsqueeze(1).repeat(1, num_gt)  # (N, M)
        repeated_y = anchors[:, 1].unsqueeze(1).repeat(1, num_gt)  # (N, M)

        # is anchor centers in gt boxes, shape: [N, M]
        dx = repeated_x - gt_boxes[:, 0]
        dy = repeated_y - gt_boxes[:, 1]

        # rotate the point in the opposite direction of box
        angle_cos = torch.cos(-gt_boxes[:, 6])
        angle_sin = torch.sin(-gt_boxes[:, 6])
        rot_x = (dx * angle_cos + dy * (-angle_sin)).abs()
        rot_y = (dx * angle_sin + dy * angle_cos).abs()

        is_in_gts = (rot_x < (gt_boxes[:, 3] / 2 + MARGIN)) & (
            rot_y < (gt_boxes[:, 4] / 2 + MARGIN))
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        is_in_cts = (rot_x < self.center_radius) & (rot_y < self.center_radius)
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [N]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :]
                                   & is_in_cts[is_in_gts_or_centers, :])

        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost)
        # select candidate topk ious for dynamic-k calculation
        topk = min(pairwise_ious.shape[0], self.topk)
        assert topk > 0
        topk_ious, _ = torch.topk(pairwise_ious, topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx],
                                    k=dynamic_ks[gt_idx].item(),
                                    largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :],
                                              dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
