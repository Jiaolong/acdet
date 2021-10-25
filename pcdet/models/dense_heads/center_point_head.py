import copy
import numpy as np
import torch
from torch import nn
from pcdet.utils.center_point_bbox_coder import CenterPointBBoxCoder
from pcdet.utils.gaussian import draw_heatmap_gaussian,gaussian_radius
from pcdet.ops.iou3d_nms.iou3d_nms_utils import circle_nms
from pcdet.utils.gaussian_focal_loss import GaussianFocalLoss
from pcdet.utils.loss_utils import WeightedL1Loss
import matplotlib.pyplot as plt


class SeparateHead(nn.Module):
    """SeparateHead for CenterHead.

    Args:
        in_channels (int): Input channels for conv_layer.
        heads (dict): Conv information.
        head_conv (int): Output channels.
            Default: 64.
        final_kernal (int): Kernal size for the last conv layer.
            Deafult: 1.
        init_bias (float): Initial bias. Default: -2.19.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=3,
                 init_bias=-2.19):
        super(SeparateHead, self).__init__()
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.extend([
                    nn.Conv2d(c_in,head_conv,kernel_size=final_kernel,stride=1,padding=final_kernel//2,bias=False),
                    nn.BatchNorm2d(head_conv,eps=0.001, momentum=0.03),
                    nn.ReLU(inplace=True)])
                c_in = head_conv

            conv_layers.append(
                nn.Conv2d(head_conv,classes,kernel_size=final_kernel,stride=1,padding=final_kernel//2,bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)


    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 2, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


class CenterHead(nn.Module):
    """CenterHead for CenterPoint.

    Args:
        mode (str): Mode of the head. Default: '3d'.
        in_channels (list[int] | int): Channels of the input feature map.
            Default: [128].
        tasks (list[dict]): Task information including class number
            and class names. Default: None.
        dataset (str): Name of the dataset. Default: 'nuscenes'.
        weight (float): Weight for location loss. Default: 0.25.
        code_weights (list[int]): Code weights for location loss. Default: [].
        common_heads (dict): Conv information for common heads.
            Default: dict().
        loss_cls (dict): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int): Output channels for share_conv_layer.
            Default: 64.
        num_heatmap_convs (int): Number of conv layers for heatmap conv layer.
            Default: 2.
        conv_cfg (dict): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 model_cfg,
                 input_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 separate_head=dict(init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 norm_bbox=True):
        super(CenterHead, self).__init__()

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.in_channels = input_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        self.model_cfg=model_cfg

        self.loss_cls = self.build_cls_loss()
        self.loss_bbox = self.build_reg_loss()
        self.bbox_coder = self.build_bbox_coder(bbox_coder)
        self.forward_ret_dict = { }


        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels,share_conv_channel,kernel_size=3,padding=1,bias=False,stride=1),
            nn.BatchNorm2d(share_conv_channel,eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True))

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                inchannels=share_conv_channel, heads=heads)
            self.task_heads.append(self.build_head(separate_head))

    def forward(self, data_dict):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        # ret_dicts = []
        x = data_dict['spatial_features_2d']
        x = self.shared_conv(x)
        self.forward_ret_dict['pred_dict_list']=[]
        for task_id, task in enumerate(self.task_heads):
            self.forward_ret_dict['pred_dict_list'].append(task(x))

        if self.training:
            self.forward_ret_dict['gt_boxes_with_class']=data_dict['gt_boxes']

        if not self.training:

            data_dict['batch_predict_list']=self.get_bboxes()

        return data_dict



    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor): Mask of the feature map with the shape
                of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_targets(self,gt_bboxes_label_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        # print("gt boxes type is ",type(gt_bboxes_label_3d))
        # print("gt boxes shape is ",gt_bboxes_label_3d.shape)
        batch_size=gt_bboxes_label_3d.shape[0]
        heatmaps, anno_boxes, inds, masks=[],[],[],[]
        for i in range(batch_size):
            cur_gt = gt_bboxes_label_3d[i]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            heatmap, anno_box, ind, mask=self.get_targets_single(cur_gt[:,:-1], cur_gt[:,-1])
            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)

        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # transpose anno_boxes
        anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # transpose inds
        inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]
        # transpose inds
        masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        # print("gt boxes shape is ",gt_bboxes_3d.shape)
        max_objs = 100
        # grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        grid_size = (pc_range[3:6] - pc_range[0:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 1
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m]- flag2)
            # print("task class is ",task_class)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            # print("task boxeses is ",task_boxes)
            # print("task classes is ",task_classes)
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 8),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            assert task_boxes[idx].shape[0]<max_objs,"task boxes num must less than max_objs"
            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=0.1)
                    radius = max(2, int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    # import pdb;pdb.set_trace()
                    # vx, vy = task_boxes[idx][k][7:]
                    # vx, vy = task_boxes[idx][k][-2:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot+np.pi).unsqueeze(0),
                        torch.cos(rot+np.pi).unsqueeze(0),
                        # vx.unsqueeze(0),
                        # vy.unsqueeze(0)
                    ])
            # heatmap_np=heatmap.permute(1,2,0).contiguous().cpu().numpy()
            # heatmap_np=heatmap_np[:,::-1,::-1]
            # plt.imshow(heatmap_np)
            # plt.show()

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks


    def get_loss(self):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            self.forward_ret_dict['gt_boxes_with_class'])
        # import pdb;pdb.set_trace()
        loss_dict = dict()
        loss_rpn=0

        for task_id, preds_dict in enumerate(self.forward_ret_dict['pred_dict_list']):
            # heatmap focal loss
            preds_dict['heatmap'] = clip_sigmoid(preds_dict['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            # import pdb;pdb.set_trace()
            loss_heatmap = self.loss_cls(
                preds_dict['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            loss_heatmap=loss_heatmap*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            loss_rpn=loss_rpn+loss_heatmap

            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict['anno_box'] = torch.cat(
                (preds_dict['reg'], preds_dict['height'],
                 preds_dict['dim'], preds_dict['rot'],),
                 # preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            code_weights=mask.new_tensor(code_weights).reshape(1,1,-1)
            bbox_weights = mask * code_weights
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights)
            loss_bbox=loss_bbox.sum()/(num+1e-4)
            loss_bbox=loss_bbox*self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap.item()
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox.item()
            loss_dict[f'task{task_id}.loss_rpn']=loss_heatmap.item()+loss_bbox.item()
            loss_rpn=loss_rpn+loss_bbox
        return  loss_rpn, loss_dict

    def get_bboxes(self):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(self.forward_ret_dict['pred_dict_list']):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict['heatmap'].shape[0]
            batch_heatmap = preds_dict['heatmap'].sigmoid()

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)


            ret_task = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                centers = boxes3d[:, [0, 1]]
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                keep = torch.tensor(
                    circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.train_cfg['min_radius'][task_id],
                        post_max_size=self.train_cfg['post_max_size']),
                    dtype=torch.long,
                    device=boxes.device)

                boxes3d = boxes3d[keep]
                scores = scores[keep]
                labels = labels[keep]
                ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                ret_task.append(ret)
            rets.append(ret_task)

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list


    def build_bbox_coder(self,coder_config):
        center_bbox_coder=CenterPointBBoxCoder(
            pc_range=coder_config['point_cloud_range'],
            out_size_factor=coder_config['out_size_factor'],
            voxel_size=coder_config['voxel_size'],
            post_center_range=coder_config['post_center_range'],
            max_num=coder_config['max_num'],
            score_threshold=coder_config['score_threshold'],
            code_size=coder_config['code_size']
        )
        return center_bbox_coder

    def build_head(self,head_config):
        separte_head=SeparateHead(
            in_channels=head_config['inchannels'],
            heads=head_config['heads'],
            head_conv=head_config['head_conv'],
            final_kernel=head_config['final_kernel'],
            init_bias=head_config['init_bias']
        )
        return separte_head

    def build_cls_loss(self):
        gaussian_focal_loss=GaussianFocalLoss(
            reduction='mean',
            loss_weight=1.0)
        return gaussian_focal_loss
    def build_reg_loss(self):
        l1_loss=WeightedL1Loss()
        return  l1_loss

def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


