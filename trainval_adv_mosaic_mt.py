# coding:utf-8
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss
from model.ema.optim_weight_ema import WeightEMA
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.utils.parser_func import parse_args, set_dataset_args
from model.utils.source_similar_pesudo_label import obtain_label
import matplotlib
import matplotlib.pyplot as plt
import pdb
import random
from roi_data_layer.minibatch_mosaic import generate_mosaic_img_train

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    cfg.MAX_NUM_GT_BOXES = 40
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    # source dataset


    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)

    print('{:d} source roidb entries'.format(len(roidb)))
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.log_ckpt_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    sampler_batch_t = sampler(train_size_t, args.batch_size)

    dataset_s = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                               imdb.num_classes, training=True)

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size,
                                               sampler=sampler_batch, num_workers=args.num_workers)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb.num_classes, training=True)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size,
                                               sampler=sampler_batch_t, num_workers=args.num_workers)
    # initilize the tensor holder here.
    im_data_s = torch.FloatTensor(1)
    im_data_w = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data_s = im_data_s.cuda()
        im_data_w = im_data_w.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data_s = Variable(im_data_s)
    im_data_w = Variable(im_data_w)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    from model.faster_rcnn.vgg16_adv import vgg16
    # from model.faster_rcnn.resnet_HTCN import resnet

    if args.net == 'vgg16':
        student_fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=False,
                           gc=False, la_attention = False, mid_attention = False)
        teacher_fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, lc=False,
                           gc=False, la_attention = False, mid_attention = False)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc, la_attention = args.LA_ATT, mid_attention = args.MID_ATT)

    else:
        print("network is not defined")
        # pdb.set_trace()

    student_fasterRCNN.create_architecture()
    teacher_fasterRCNN.create_architecture()


    print("load pretrain checkpoint %s" % (args.load_name))
    checkpoint = torch.load(args.load_name)
    student_fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load pretrain model successfully!')

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    student_detection_params = []
    params = []
    for key, value in dict(student_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            student_detection_params += [value]

    teacher_detection_params = []
    for key, value in dict(teacher_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            teacher_detection_params += [value]
            value.requires_grad = False

    if args.optimizer == "adam":
        lr = lr * 0.1
        student_optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        student_optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    teacher_optimizer = WeightEMA(
        teacher_detection_params, student_detection_params, alpha=args.teacher_alpha
    )

    if args.cuda:
        student_fasterRCNN.cuda()
        teacher_fasterRCNN.cuda()

    if args.resume:
        student_checkpoint = torch.load(args.student_load_name)
        args.session = student_checkpoint["session"]
        args.start_epoch = student_checkpoint["epoch"]
        student_fasterRCNN.load_state_dict(student_checkpoint["model"])
        student_optimizer.load_state_dict(student_checkpoint["optimizer"])
        lr = student_optimizer.param_groups[0]["lr"]
        if "pooling_mode" in student_checkpoint.keys():
            cfg.POOLING_MODE = student_checkpoint["pooling_mode"]
        print("loaded student checkpoint %s" % (args.student_load_name))

        teacher_checkpoint = torch.load(args.teacher_load_name)
        teacher_fasterRCNN.load_state_dict(teacher_checkpoint["model"])
        if "pooling_mode" in teacher_checkpoint.keys():
            cfg.POOLING_MODE = teacher_checkpoint["pooling_mode"]
        print("loaded teacher checkpoint %s" % (args.teacher_load_name))

    if args.mGPUs:
        student_fasterRCNN = nn.DataParallel(student_fasterRCNN)
        teacher_fasterRCNN = nn.DataParallel(teacher_fasterRCNN)

    iters_per_epoch = int(10000 / args.batch_size)

    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")
        
    count_iter = 0
    all_gt_boxes_padding = {}
    all_num_boxes_cpu = {}
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        student_fasterRCNN.train()
        teacher_fasterRCNN.eval()

        count_step = 0
        loss_temp_last = 1  
        loss_temp = 0
        loss_rpn_cls_temp = 0
        loss_rpn_box_temp = 0
        loss_rcnn_cls_temp = 0
        loss_rcnn_box_temp = 0

        start = time.time()
        if epoch - 1 in  args.lr_decay_step:
            adjust_learning_rate(student_optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(dataloader_s)
        data_iter_t = iter(dataloader_t)

        for step in range(1, iters_per_epoch + 1):
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(dataloader_s)
                data_s = next(data_iter_s)
            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(dataloader_t)
                data_t = next(data_iter_t)


            random_indexs = [random.randint(0, len(dataset_s) - 1) for _ in range(3)]
            random3img = [dataset_s[random_index] for random_index in random_indexs]
            all_img = [data_s] + random3img
            w0, h0 = data_s[0].shape[4], data_s[0].shape[3]




            gt_boxes_all = []
            input_all = []
            for idx in range(len(all_img)):

                data_s = all_img[idx]
                if idx != 0:

                    data_ss = torch.unsqueeze(data_s[0], 0)[:, 1, :, :, :]
                    data_sw = torch.unsqueeze(data_s[0], 0)[:, 0, :, :, :]
                    im_data_w.resize_(data_sw.size()).copy_(data_sw)
                    im_info.resize_(torch.unsqueeze(data_s[1], 0).size()).copy_(torch.unsqueeze(data_s[1], 0))
                    input_all.append(data_ss)
                else:

                    data_ss = data_s[0][:, 1, :, :, :]
                    data_sw = data_s[0][:, 0, :, :, :]
                    im_data_w.resize_(data_sw.size()).copy_(data_sw)
                    im_info.resize_(data_s[1].size()).copy_(data_s[1])
                    input_all.append(data_ss)


                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()


                teacher_fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, out_d_pixel, out_d = teacher_fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes)


                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]


                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = (
                                    box_deltas.view(-1, 4)
                                    * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS
                            ).cuda()
                                    + torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_MEANS
                            ).cuda()
                            )
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = (
                                    box_deltas.view(-1, 4)
                                    * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS
                            ).cuda()
                                    + torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_MEANS
                            ).cuda()
                            )
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb_t.classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))


                scores = scores.squeeze()
                pred_boxes = pred_boxes.squeeze()
                pre_thresh = 0.0
                gt_boxes_idx = []
                thresh = args.threshold
                empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
                for j in range(1, len(imdb_t.classes)):
                    inds = torch.nonzero(scores[:, j] > pre_thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                        cls_dets = cls_dets[order]
                        keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]

                        cls_dets_numpy = cls_dets.cpu().numpy()

                        for i in range(np.minimum(10, cls_dets_numpy.shape[0])):
                            bbox = tuple(
                                int(np.round(x)) for x in cls_dets_numpy[i, :4]
                            )
                            score = cls_dets_numpy[i, -1]
                            if score > thresh:
                                gt_boxes_idx.append(list(bbox[0:4]) + [j])
                gt_boxes_all.append(gt_boxes_idx)

            mosaic_img, gt_boxes_mosaic = generate_mosaic_img_train(input_all, gt_boxes_all, w0, h0)


            gt_boxes_padding = torch.FloatTensor(cfg.MAX_NUM_GT_BOXES, 5).zero_()
            if len(gt_boxes_mosaic) != 0:
                gt_boxes_numpy = torch.FloatTensor(gt_boxes_mosaic)
                num_boxes_cpu = torch.LongTensor(
                    [min(gt_boxes_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
                )
                gt_boxes_padding[:num_boxes_cpu, :] = gt_boxes_numpy[:num_boxes_cpu]
            else:
                num_boxes_cpu = torch.LongTensor([0])

            gt_boxes_padding = torch.unsqueeze(gt_boxes_padding, 0)
            gt_boxes.resize_(gt_boxes_padding.size()).copy_(gt_boxes_padding)
            num_boxes.resize_(num_boxes_cpu.size()).copy_(num_boxes_cpu)
            im_data_s.resize_(mosaic_img.size()).copy_(mosaic_img)

            im_info.resize_(all_img[0][1].size()).copy_(all_img[0][1])


            student_fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d = student_fasterRCNN(im_data_s, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()


            count_step += 1

            loss_temp += loss.item()
            loss_rpn_cls_temp += rpn_loss_cls.mean().item()
            loss_rpn_box_temp += rpn_loss_box.mean().item()
            loss_rcnn_cls_temp += RCNN_loss_cls.mean().item()
            loss_rcnn_box_temp += RCNN_loss_bbox.mean().item()


            ######################### da loss 1 #####################################
            # domain label
            domain_s = Variable(torch.zeros(out_d.size(0)).long().cuda())
            # global alignment loss
            dloss_s = 0.5 * FL(out_d, domain_s)


            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_s_p = 0.5 * torch.mean(out_d_pixel ** 2)

            #put target data into variable
            data_t[0] = data_t[0][0, 0, :, :, :].unsqueeze(0)
            im_data_w.resize_(data_t[0].size()).copy_(data_t[0])
            im_info.resize_(data_t[1].size()).copy_(data_t[1])
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

            out_d_pixel, out_d = student_fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes, target=True)


            ######################### da loss 1 #####################################
            # domain label
            domain_t = Variable(torch.ones(out_d.size(0)).long().cuda())
            dloss_t = 0.5 * FL(out_d, domain_t)


            ######################### da loss 3 #####################################
            # local alignment loss
            dloss_t_p = 0.5 * torch.mean((1 - out_d_pixel) ** 2)




            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            teacher_fasterRCNN.zero_grad()

            if step % 2500 == 0:
                teacher_optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()

                loss_temp /= count_step
                loss_rpn_cls_temp /= count_step
                loss_rpn_box_temp /= count_step
                loss_rcnn_cls_temp /= count_step
                loss_rcnn_box_temp /= count_step


                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    dloss_s = dloss_s.item()
                    dloss_t = dloss_t.item()
                    dloss_s_p = dloss_s_p.item()
                    dloss_t_p = dloss_t_p.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e, step: %3d, count: %3d" \
                      % (args.session, epoch, step, iters_per_epoch, loss_temp, lr, count_step, count_iter))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f dloss s: %.4f dloss t: %.4f dloss_s_p: %.4f dloss_t_p: %.4f eta: %.4f" \
                    % (loss_rpn_cls_temp, loss_rpn_box_temp, loss_rcnn_cls_temp, loss_rcnn_box_temp, dloss_s, dloss_t, dloss_s_p, dloss_t_p,
                       args.eta))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls_temp,
                        'loss_rpn_box': loss_rpn_box_temp,
                        'loss_rcnn_cls': loss_rcnn_cls_temp,
                        'loss_rcnn_box': loss_rcnn_box_temp
                    }
                    # logger.add_scalars("logs_s_{}/losses".format(args.session), info,
                    #                    (epoch - 1) * iters_per_epoch + step)
                    logger.add_scalars(args.log_ckpt_name, info,
                                       (epoch - 1) * iters_per_epoch + step)

                count_step = 0
                loss_temp_last = loss_temp
                loss_temp = 0
                loss_rpn_cls_temp = 0
                loss_rpn_box_temp = 0
                loss_rcnn_cls_temp = 0
                loss_rcnn_box_temp = 0

                start = time.time()


        student_save_name = os.path.join(output_dir, 'student',
                                         'lg_2500_target_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                             args.dataset_t,
                                             args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': student_fasterRCNN.module.state_dict() if args.mGPUs else student_fasterRCNN.state_dict(),
            'optimizer': student_optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, student_save_name)
        print('save student model: {}'.format(student_save_name))

        teacher_save_name = os.path.join(output_dir, 'teacher',
                                 'lg_2500_target_{}_eta_{}_local_{}_global_{}_gamma_{}_session_{}_epoch_{}_step_{}.pth'.format(
                                     args.dataset_t,args.eta,
                                     args.lc, args.gc, args.gamma,
                                     args.session, epoch,
                                     step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': teacher_fasterRCNN.module.state_dict() if args.mGPUs else teacher_fasterRCNN.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, teacher_save_name)
        print('save model: {}'.format(teacher_save_name))

    if args.use_tfboard:
        logger.close()
