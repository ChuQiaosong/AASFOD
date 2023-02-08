# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
#from scipy.misc import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import pdb
from .randaugment import RandAugment
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pdb
import torch

def get_minibatch(roidb, num_classes, random3db, seg_return=False, is_training=False):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales, gt_boxes_all = _get_image_blob(roidb, random_scale_inds, random3db, is_training=is_training)

  blobs = {'data': im_blob}

  # assert len(im_scales) == 1, "Single batch only"
  # assert len(roidb) == 1, "Single batch only"

  gt_boxes_all[:, 0:4] = gt_boxes_all[:, 0:4] * im_scales[0]

  blobs['gt_boxes'] = gt_boxes_all
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)
  if seg_return:
    blobs['seg_map'] = roidb[0]['seg_map']
  blobs['img_id'] = roidb[0]['img_id']
  blobs['path'] = roidb[0]['image']

  blobs['flipped'] = roidb[0]['flipped']

  return blobs


def _get_image_blob(roidb, scale_inds, random3db, is_training=False):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """

  num_images = len(roidb)

  processed_ims = []
  im_scales = []

  for i in range(num_images):



    img = Image.open(roidb[i]['image'])


    im = np.array(img).astype(np.float32)
    if len(im.shape) == 2:
      im = im[:, :, np.newaxis]
      im = np.concatenate((im, im, im), axis=2)
    im = im[:, :, ::-1]



    h0, w0 = im.shape[0], im.shape[1]

    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

    if is_training:


      mosaic_img, gt_boxes_all = generate_mosaic_img(roidb[i], random3db, w0, h0)
      im_mosaic = np.array(mosaic_img).astype(np.float32)
      if len(im_mosaic.shape) == 2:
        im_mosaic = im_mosaic[:, :, np.newaxis]
        im_mosaic = np.concatenate((im_mosaic, im_mosaic, im_mosaic), axis=2)
      im_mosaic = im_mosaic[:, :, ::-1]

      target_size = cfg.TRAIN.SCALES[scale_inds[i]]
      im, im_scale = prep_im_for_blob(im_mosaic, cfg.PIXEL_MEANS, target_size,
                      cfg.TRAIN.MAX_SIZE)
      im_scales.append(im_scale)
      processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, gt_boxes_all


def fill_truth_detection(bboxes, dx, dy, sx, sy, net_w, net_h):

  np.random.shuffle(bboxes)
  bboxes[:, 0] -= dx
  bboxes[:, 2] -= dx
  bboxes[:, 1] -= dy
  bboxes[:, 3] -= dy

  bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
  bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

  bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
  bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

  out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                          ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                          ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                          ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
  list_box = list(range(bboxes.shape[0]))
  for i in out_box:
    list_box.remove(i)
  bboxes = bboxes[list_box]


  bboxes[:, 0] *= (net_w / sx)
  bboxes[:, 2] *= (net_w / sx)
  bboxes[:, 1] *= (net_h / sy)
  bboxes[:, 3] *= (net_h / sy)

  return bboxes

def rect_intersection(a, b):
  minx = max(a[0], b[0])
  miny = max(a[1], b[1])

  maxx = min(a[2], b[2])
  maxy = min(a[3], b[3])
  return [minx, miny, maxx, maxy]

def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight):

  img = mat
  oh, ow, _ = img.shape
  pleft, ptop, swidth, sheight = int(pleft), int(ptop), int(swidth), int(sheight)
  # crop
  src_rect = [pleft, ptop, swidth + pleft, sheight + ptop]  # x1,y1,x2,y2
  img_rect = [0, 0, ow, oh]
  new_src_rect = rect_intersection(src_rect, img_rect)

  dst_rect = [max(0, -pleft), max(0, -ptop), max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
              max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]
  # cv2.Mat sized

  if (src_rect[0] == 0 and src_rect[1] == 0 and src_rect[2] == img.shape[0] and src_rect[3] == img.shape[1]):
    pil_img = Image.fromarray(np.uint8(img))
    resized_img = pil_img.resize((w, h))
    sized = np.array(resized_img)
  else:
    cropped = np.zeros([sheight, swidth, 3])
    cropped[:, :, ] = np.mean(img, axis=(0, 1))

    cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
      img[new_src_rect[1]:new_src_rect[3], new_src_rect[0]:new_src_rect[2]]

    # resize (h0,w0)
    pil_img = Image.fromarray(np.uint8(cropped))
    resized_img = pil_img.resize((w, h))
    sized = np.array(resized_img)

    return sized

def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
  bboxes[:, 0] -= dx
  bboxes[:, 2] -= dx
  bboxes[:, 1] -= dy
  bboxes[:, 3] -= dy

  bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
  bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

  bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
  bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

  out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                          ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                          ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                          ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
  list_box = list(range(bboxes.shape[0]))
  for i in out_box:
    list_box.remove(i)
  bboxes = bboxes[list_box]

  bboxes[:, 0] += xd
  bboxes[:, 2] += xd
  bboxes[:, 1] += yd
  bboxes[:, 3] += yd

  return bboxes

def blend_truth_mosaic(out_img, img, bboxes, w, h, cut_x, cut_y, i_mixup,
                         left_shift, right_shift, top_shift, bot_shift):

  left_shift = min(left_shift, w - cut_x)
  top_shift = min(top_shift, h - cut_y)
  right_shift = min(right_shift, cut_x)
  bot_shift = min(bot_shift, cut_y)


  if i_mixup == 0:
    if bboxes.shape[0] != 0:
      bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
    out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
  if i_mixup == 1:
    if bboxes.shape[0] != 0:
      bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, w - cut_x, cut_y, cut_x, 0)
    out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:w - right_shift]
  if i_mixup == 2:
    if bboxes.shape[0] != 0:
      bboxes = filter_truth(bboxes, left_shift, cut_y - bot_shift, cut_x, h - cut_y, 0, cut_y)
    out_img[cut_y:, :cut_x] = img[cut_y - bot_shift:h - bot_shift, left_shift:left_shift + cut_x]
  if i_mixup == 3:
    if bboxes.shape[0] != 0:
      bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bot_shift, w - cut_x, h - cut_y, cut_x, cut_y)
    out_img[cut_y:, cut_x:] = img[cut_y - bot_shift:h - bot_shift, cut_x - right_shift:w - right_shift]

  return out_img, bboxes

def generate_mosaic_img(roidb, random3db, w0, h0):


  min_offset = 0.2
  cut_x = random.randint(int(w0 * min_offset), int(w0 * (1 - min_offset)))
  cut_y = random.randint(int(h0 * min_offset), int(h0 * (1 - min_offset)))


  out_img = np.zeros([h0, w0, 3])
  gt_boxes_all = []


  all_roidb = [roidb] + random3db

  for idx in range(len(all_roidb)):  # idx = 0,1,2,3

    tmp_db = all_roidb[idx]
    img = Image.open(tmp_db['image'])   #(h,w,rgb)
    img = np.array(img).astype(np.float32)



    oh, ow, oc = img.shape
    dh, dw, dc = np.array(np.array([oh, ow, oc]) * 0.2, dtype=np.int)
    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot


    if cfg.TRAIN.USE_ALL_GT:
      # Include all ground truth boxes
      gt_inds = np.where(tmp_db['gt_classes'] != 0)[0]
    else:
      # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
      gt_inds = np.where((tmp_db['gt_classes'] != 0) & np.all(tmp_db['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = tmp_db['boxes'][gt_inds, :]
    gt_boxes[:, 4] = tmp_db['gt_classes'][gt_inds]

    bboxes = gt_boxes.copy()
    truth, min_w_h = fill_truth_detection(bboxes, pleft, ptop, swidth, sheight, w0, h0)

    ai = image_data_augmentation(img, w0, h0, pleft, ptop, swidth, sheight)

    left_shift = int(min(cut_x, max(0, (-int(pleft) * w0 / swidth))))
    top_shift = int(min(cut_y, max(0, (-int(ptop) * h0 / sheight))))

    right_shift = int(min((w0 - cut_x), max(0, (-int(pright) * w0 / swidth))))
    bot_shift = int(min(h0 - cut_y, max(0, (-int(pbot) * h0 / sheight))))

    out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth, w0, h0, cut_x,
                                           cut_y, idx, left_shift, right_shift, top_shift, bot_shift)
    gt_boxes_all.append(out_bbox)
  gt_boxes_all = np.concatenate(gt_boxes_all, axis=0)

  return out_img, gt_boxes_all


def generate_mosaic_img_train(input_all, gt_boxes_all, w0, h0):


  min_offset = 0.2
  cut_x = random.randint(int(w0 * min_offset), int(w0 * (1 - min_offset)))
  cut_y = random.randint(int(h0 * min_offset), int(h0 * (1 - min_offset)))


  out_img = np.zeros([h0, w0, 3])
  gt_boxes_mosaic = []



  for idx in range(len(input_all)):  # idx = 0,1,2,3

    # NCHW -> CHW -> HWC(h,w,rgb)
    img = input_all[idx][0].permute(1,2,0)
    img = img.numpy().astype(np.float32)



    oh, ow, oc = img.shape
    dh, dw, dc = np.array(np.array([oh, ow, oc]) * 0.2, dtype=np.int)
    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot


    gt_boxes = np.array(gt_boxes_all[idx], dtype=np.float32)
    bboxes = gt_boxes.copy()


    if bboxes.shape[0] == 0:
      truth = bboxes
    else:
      truth = fill_truth_detection(bboxes, pleft, ptop, swidth, sheight, w0, h0)


    ai = image_data_augmentation(img, w0, h0, pleft, ptop, swidth, sheight)

    left_shift = int(min(cut_x, max(0, (-int(pleft) * w0 / swidth))))
    top_shift = int(min(cut_y, max(0, (-int(ptop) * h0 / sheight))))

    right_shift = int(min((w0 - cut_x), max(0, (-int(pright) * w0 / swidth))))
    bot_shift = int(min(h0 - cut_y, max(0, (-int(pbot) * h0 / sheight))))


    out_img, out_bbox = blend_truth_mosaic(out_img, ai, truth, w0, h0, cut_x,
                                           cut_y, idx, left_shift, right_shift, top_shift, bot_shift)
    if out_bbox.shape[0] != 0:
      gt_boxes_mosaic.append(out_bbox)
  if len(gt_boxes_mosaic) > 0:
    gt_boxes_mosaic = np.concatenate(gt_boxes_mosaic, axis=0)
  out_img = torch.from_numpy(out_img).permute(2,0,1) # HWC -> CHW
  out_img = torch.unsqueeze(out_img, 0)

  return out_img, gt_boxes_mosaic

