import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_crop.modules.roi_crop import _RoICrop
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta,grad_reverse, local_attention, middle_attention
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,lc,gc, la_attention = False, mid_attention = False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        self.la_attention = la_attention
        self.mid_attention = mid_attention
        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0, 0)

        # self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()

    def obtain_pseudo_labels(self, im_data, im_info, gt_boxes, num_boxes, thresh=0.7):

        base_feat1 = self.RCNN_base1(im_data)
        base_feat2 = self.RCNN_base2(base_feat1)
        base_feat = self.RCNN_base3(base_feat2)

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        rois = Variable(rois)
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)

        cls_prob = F.softmax(cls_score, 1)


        cls_prob = cls_prob.view(1, rois.size(1), -1)
        bbox_pred = bbox_pred.view(1, rois.size(1), -1)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # thresh = 0.7
        max_per_image = 15

        all_boxes = []

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * self.n_classes)

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # pred_boxes /= im_info[0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()


        record_scores = scores
        all_scores = []

        for j in range(1, self.n_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]

                keep = nms(cls_dets, cls_scores[order], cfg.TEST.NMS)

                cls_dets = cls_dets[keep.view(-1).long()]

                cls_dets = torch.cat((cls_dets, torch.ones(len(cls_dets)).unsqueeze(1).cuda() * j), 1)

                all_boxes.append(cls_dets)

                all_scores.append(record_scores[inds, :][keep.view(-1).long()])

        # breakpoint()

        if len(all_boxes):

            all_all_boxes = torch.cat(all_boxes, dim=0)
            all_all_scores = torch.cat(all_scores, dim=0)

            if max_per_image > 0:
                # all_all_boxes = torch.cat(all_boxes, dim=0)
                # image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self.n_classes)])
                if len(all_all_boxes) > max_per_image:
                    scores = all_all_boxes[:, -2]

                    image_thresh = torch.sort(scores)[0][-max_per_image]
                    mask = torch.nonzero(scores > image_thresh, as_tuple=True)
                    all_all_scores = all_all_scores[mask]
                    all_all_boxes = all_all_boxes[scores > image_thresh]

            return torch.cat((all_all_boxes[:, :4], all_all_boxes[:, -1].unsqueeze(1)), 1).unsqueeze(0), all_all_scores
        else:
            return [], []

    def augment_pseudo_labels(self, rois, im_data_s, im_info, gt_boxes, num_boxes, thresh=0.7):



        base_feat1_s = self.RCNN_base1(im_data_s)
        base_feat2_s = self.RCNN_base2(base_feat1_s)
        base_feat_s = self.RCNN_base3(base_feat2_s)

        pooled_feat = self.RCNN_roi_align(base_feat_s, rois.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        cls_score = self.RCNN_cls_score(pooled_feat)

        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        cls_prob = cls_prob.view(1, rois.size(1), -1)
        bbox_pred = bbox_pred.view(1, rois.size(1), -1)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # thresh = 0.7
        max_per_image = 15

        all_boxes = []

        if cfg.TEST.BBOX_REG:
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * self.n_classes)

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        # pred_boxes /= im_info[0][2].item()

        scores = scores.squeeze()         #(300,9)
        pred_boxes = pred_boxes.squeeze()

        record_scores = scores
        all_scores = []
        for j in range(1, self.n_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]

                keep = nms(cls_dets, cls_scores[order], cfg.TEST.NMS)

                cls_dets = cls_dets[keep.view(-1).long()]

                cls_dets = torch.cat((cls_dets, torch.ones(len(cls_dets)).unsqueeze(1).cuda() * j), 1)

                all_boxes.append(cls_dets)

                all_scores.append(record_scores[inds, :][keep.view(-1).long()])

        # breakpoint()

        if len(all_boxes):

            all_all_boxes = torch.cat(all_boxes, dim=0)
            all_all_scores = torch.cat(all_scores, dim=0)

            if max_per_image > 0:
                # all_all_boxes = torch.cat(all_boxes, dim=0)
                # image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, self.n_classes)])
                if len(all_all_boxes) > max_per_image:
                    scores = all_all_boxes[:, -2]  #-1是j，-2是分数  # N x 1

                    image_thresh = torch.sort(scores)[0][-max_per_image]
                    mask = torch.nonzero(scores > image_thresh, as_tuple=True)
                    all_all_scores = all_all_scores[mask]
                    all_all_boxes = all_all_boxes[scores > image_thresh]

            return torch.cat((all_all_boxes[:, :4], all_all_boxes[:, -1].unsqueeze(1)), 1).unsqueeze(0), all_all_scores
        else:
            return [], []

    def filter_pseudo_labels(self, im_data, pseudo_gt, thresh=0.7):

        # breakpoint()
        pseudo_gt = pseudo_gt.squeeze(0)

        nt = len(pseudo_gt)
        box_label = torch.zeros(nt, 1).cuda()
        rois = torch.cat((box_label, pseudo_gt[:, :4].clone()), dim=-1)

        num_image = im_data.size(0)
        all_vals, all_idx = [], []
        for i in range(num_image):
            base_feat = self.RCNN_base(im_data[i, :, :, :].unsqueeze(0))

            # breakpoint()

            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat = self._head_to_tail(pooled_feat)

            score = self.RCNN_cls_score(pooled_feat)

            val, idx = torch.max(score, dim=1, keepdim=True)
            all_vals.append(val)
            all_idx.append(idx)

        # inds = torch.nonzero(score[:,j]>thresh).view(-1)
        all_vals = torch.cat(all_vals, dim=1)
        all_idx = torch.cat(all_idx, dim=1)

        idx_first = pseudo_gt[:, 4].unsqueeze(1).repeat((1, all_idx.size(1))).long()

        keep_idx = torch.all(all_idx == idx_first, dim=1)

        # if not all(keep_idx):
        # 	breakpoint()

        pseudo_gt = pseudo_gt[keep_idx, :]

        if len(pseudo_gt):
            pseudo_gt = pseudo_gt.unsqueeze(0)
        else:
            pseudo_gt = []

        return pseudo_gt

    def forward(self, im_data, im_info, gt_boxes, num_boxes,target=False,eta=1.0):


        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data


        # feed image data to base model to obtain base feature map
        base_feat1 = self.RCNN_base1(im_data)
        d_pixel = self.netD_pixel(grad_reverse(base_feat1, lambd=eta))


        base_feat2 = self.RCNN_base2(base_feat1)


        base_feat = self.RCNN_base3(base_feat2)
        domain_p = self.netD(grad_reverse(base_feat, lambd=eta))

        if target:
            return d_pixel, domain_p


        # feed base feature map tp RPN to obtain rois


        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0


        rois = Variable(rois)
        #print("rois.shape:{}".format(rois.size()))

        # do roi pooling based on predicted rois


        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)  #(A,4096)


        #feat_pixel = torch.zeros(feat_pixel.size()).cuda()



        # compute bbox offset

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability

        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, d_pixel, domain_p#,diff

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
