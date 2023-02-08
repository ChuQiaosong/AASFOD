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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient, FocalLoss, sampler, calc_supp, EFocalLoss


from model.utils.parser_func import parse_args, set_dataset_args
from model.ema.optim_weight_ema import WeightEMA
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.utils.loss import Entropy, score_function
from model.utils.obtain_predictions import obtain_predictions

from PIL import Image
import matplotlib.pyplot as plt
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


    # target dataset
    imdb_t, roidb_t, ratio_list_t, ratio_index_t = combined_roidb(args.imdb_name_target)
    train_size_t = len(roidb_t)
    print('{:d} target roidb entries'.format(len(roidb_t)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.log_ckpt_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch_t = sampler(train_size_t, args.batch_size)
    dataset_t = roibatchLoader(roidb_t, ratio_list_t, ratio_index_t, args.batch_size, \
                               imdb_t.num_classes, training=True)
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
    # from model.faster_rcnn.vgg16_adv import vgg16
    from model.faster_rcnn.vgg16_adv import vgg16
    # from model.faster_rcnn.resnet_HTCN import resnet

    if args.net == 'vgg16':
        # fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
        fasterRCNN = vgg16(imdb_t.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_t.classes, 101, pretrained=True, class_agnostic=args.class_agnostic,
                            lc=args.lc, gc=args.gc, la_attention = args.LA_ATT, mid_attention = args.MID_ATT)

    else:
        print("network is not defined")
        # pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load pretrain checkpoint %s" % (args.load_name)) #--load_name
    checkpoint = torch.load(args.load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load pretrain model successfully!')

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        fasterRCNN.cuda()


    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)


    iters_per_epoch = int(10000 / args.batch_size)
    if args.ef:
        FL = EFocalLoss(class_num=2, gamma=args.gamma)
    else:
        FL = FocalLoss(class_num=2, gamma=args.gamma)


    count_iter = 0
    img_paths = []
    img_paths_similar = []
    img_paths_disimilar = []
    A_score_list = []
    H_list = []
    img_paths_no_label = []
    data_iter_t = iter(dataloader_t)
    zero_count = 0

    fasterRCNN.eval()
    def apply_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    fasterRCNN.apply(apply_dropout)
    for step in range(len(dataloader_t)):

        data_t = next(data_iter_t)

        weak_aug_data = data_t[0][:, 0, :, :, :]
        im_data_w.resize_(weak_aug_data.size()).copy_(weak_aug_data)
        im_info.resize_(data_t[1].size()).copy_(data_t[1])
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        img_path = data_t[-2][0]


        T = 20
        for t in range(T):
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, out_d_pixel, out_d = fasterRCNN(im_data_w, im_info, gt_boxes, num_boxes)

            if t == 0:
                g = cls_prob
                l = bbox_pred
            g = torch.cat((g, cls_prob), dim=0)
            l = torch.cat((l, bbox_pred), dim=0)

        g_mean = torch.mean(g, dim=0)
        uc = torch.mean(torch.sum(g ** 2, dim=2), dim=0) - torch.sum(g_mean ** 2, dim=1)
        l_mean = torch.mean(l, dim=0)
        ul = torch.mean(torch.sum(l ** 2, dim=2), dim=0) - torch.sum(l_mean ** 2, dim=1)
        u = uc * ul
        image_uncertainty = torch.sum(u)
        A_score_list.append(image_uncertainty.detach().cpu().numpy())
        img_paths.append(img_path)


    A_score_list = np.array(A_score_list)

    img_paths = np.array(img_paths)

    index = np.argsort(A_score_list)

    pre_defined_threshold = 0.8  # please set your own pre_defined_threshold
    num_similar = (1 - pre_defined_threshold) * len(img_paths)
    img_paths_similar = img_paths[index][-num_similar:]
    img_paths_disimilar = img_paths[index][:-num_similar]


    # write your own path of train.txt of source_similar data
    similar_txt_path = ".../ImageSets/Main/train.txt"
    with open(similar_txt_path, "w+") as f_similar:
        for path in img_paths_similar:
            image_index = path[38:-4].strip() + '\n'
            f_similar.write(image_index)

    # write your own path of train.txt of source_dissimilar data
    disimilar_txt_path = ".../ImageSets/Main/train.txt"
    with open(disimilar_txt_path, "w+") as f_disimilar:
        for path in img_paths_disimilar:
            image_index = path[38:-4].strip() + '\n'
            f_disimilar.write(image_index)

    print("done")



