# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
from metanets import MetaOptimizer, MetaInitializer
from networks import L2normLoss, SmoothL1Loss
import numpy as np
import config
from utils import compute_filter_size, generate_anchors, adaptable_conv2d

class ROAM(nn.Module):

    def __init__(self, feat_extractor, summary_writer=None):
        super(ROAM, self).__init__()
        self.feat_extractor = feat_extractor
        self.meta_init = MetaInitializer()
        self.meta_opti = MetaOptimizer()
        self.l2loss = L2normLoss()
        self.smoothl1loss = SmoothL1Loss()
        self.writer = summary_writer

    def forward(self, patches, label_maps, gt_bboxes, iter_step=0):
        # ----------------- feature extraction -----------------------------------------
        with torch.no_grad():
            feats = self.feat_extractor(patches.contiguous().view([-1] + list(patches.shape[2:])))
        win_cell_sz = label_maps.size(3)
        feats = F.interpolate(feats, (win_cell_sz, win_cell_sz), mode='bilinear', align_corners=True)
        feats = feats.view(list(patches.shape[0:2]) + list(feats.shape[1:]))

        map_loss_total, bbox_loss_total, meta_loss_total = 0, 0, 0
        # --------------- initial frame meta updating ----------------------------------
        init_params, init_lrs = self.meta_init.initialize()
        self.meta_opti.initialize(init_lrs)
        # random scale params to prevent overfitting
        offset = (config.feat_channels + 1) * config.cf_channels + \
                 config.base_filter_size[0] * config.base_filter_size[1] * config.cf_channels + 1
        rand_scales_cf = torch.exp(config.rand_scale_radius_cf * (2 * torch.rand(offset) - 1))
        rand_scales_reg = torch.exp(config.rand_scale_radius_reg * (2 * torch.rand(len(init_params)-offset) - 1))
        rand_scales = torch.cat([rand_scales_cf, rand_scales_reg], 0)
        if torch.cuda.is_available():
            rand_scales = rand_scales.cuda()
        init_params = init_params / rand_scales

        filter_size = compute_filter_size(gt_bboxes[:, 0, 2:])
        base_anchor_sizes = filter_size/config.filter_scale*config.cell_sz
        anchors = generate_anchors(win_cell_sz, base_anchor_sizes)
        # calculate initial update loss
        n_init_aug = len(config.aug_init_scales)*len(config.aug_init_ratios)
        pred_map, pred_bbox = adaptable_conv2d(feats[:, 0:n_init_aug], init_params*rand_scales, filter_size)
        map_loss = self.l2loss(pred_map, label_maps[:, 0:n_init_aug])
        bbox_loss = self.smoothl1loss(pred_bbox, gt_bboxes[:, 0:n_init_aug], anchors)
        loss = map_loss + bbox_loss
        grads = torch.autograd.grad(loss, init_params, create_graph=True)
        updated_params = init_params - init_lrs * grads[0]

        # calculate initial meta loss
        if self.training:
            lh_idx = np.random.randint(config.look_ahead) + n_init_aug
            lh_pred_map, lh_pred_bbox = adaptable_conv2d(feats[:, lh_idx: lh_idx + 1], updated_params * rand_scales, filter_size)
            lh_map_loss = self.l2loss(lh_pred_map, label_maps[:, lh_idx:lh_idx + 1])
            lh_bbox_loss = self.smoothl1loss(lh_pred_bbox, gt_bboxes[:, lh_idx: lh_idx + 1], anchors)
        else:
            lh_pred_map, lh_pred_bbox = adaptable_conv2d(feats[:, n_init_aug:config.look_ahead + n_init_aug], updated_params * rand_scales, filter_size)
            lh_map_loss = self.l2loss(lh_pred_map, label_maps[:, n_init_aug:config.look_ahead + n_init_aug])
            lh_bbox_loss = self.smoothl1loss(lh_pred_bbox, gt_bboxes[:, n_init_aug:config.look_ahead + n_init_aug], anchors)

        meta_loss_init = lh_map_loss + lh_bbox_loss
        if self.training and iter_step % config.disp_inter == 0 and torch.cuda.current_device() == 0:
            self.writer.add_scalar('roam_training/init_update_loss', loss.data.item(), iter_step)
            self.writer.add_scalar('roam_training/init_meta_loss', meta_loss_init.data.item(), iter_step)
            self.writer.add_histogram('roam_training/init_cf_params', init_params[:offset], iter_step)
            self.writer.add_histogram('roam_training/init_cf_lrs', init_lrs[:offset], iter_step)
            self.writer.add_histogram('roam_training/init_reg_params', init_params[offset:], iter_step)
            self.writer.add_histogram('roam_training/init_reg_lrs', init_lrs[offset:], iter_step)
        print(bbox_loss.data.item(), lh_bbox_loss.data.item())
        map_loss_total += lh_map_loss
        bbox_loss_total += lh_bbox_loss
        meta_loss_total += meta_loss_init

        # --------------- subsequent frames meta updating ------------------------------
        for k in range(1, config.time_step):
            # adapt to new size
            filter_size = compute_filter_size(gt_bboxes[:, k * config.look_ahead + n_init_aug, 2:])
            base_anchor_sizes = filter_size / config.filter_scale * config.cell_sz
            anchors = generate_anchors(win_cell_sz, base_anchor_sizes)

            # calculate update loss
            training_idxes = range((k - 1) * config.look_ahead + n_init_aug, k * config.look_ahead + n_init_aug)
            pred_map, pred_bbox = adaptable_conv2d(feats[:, training_idxes], updated_params * rand_scales, filter_size)
            map_loss = self.l2loss(pred_map, label_maps[:, training_idxes])
            bbox_loss = self.smoothl1loss(pred_bbox, gt_bboxes[:, training_idxes], anchors)
            loss = map_loss + bbox_loss
            # meta update
            grads = torch.autograd.grad(loss, updated_params, retain_graph=True)
            updated_params = self.meta_opti.meta_update(updated_params, loss, grads[0], self.writer, iter_step)

            # calculate meta loss
            if self.training:
                delta_lh_idx = np.random.randint(config.look_ahead)
                lh_idx = k * config.look_ahead + n_init_aug + delta_lh_idx
                lh_pred_map, lh_pred_bbox = adaptable_conv2d(feats[:, lh_idx: lh_idx + 1], updated_params * rand_scales, filter_size)
                lh_map_loss = self.l2loss(lh_pred_map, label_maps[:, lh_idx:lh_idx + 1])
                lh_bbox_loss = self.smoothl1loss(lh_pred_bbox, gt_bboxes[:, lh_idx:lh_idx + 1], anchors)
            else:
                chosen_idxes = range(k * config.look_ahead + n_init_aug, (k + 1) * config.look_ahead + n_init_aug)
                lh_pred_map, lh_pred_bbox= adaptable_conv2d(feats[:, chosen_idxes], updated_params * rand_scales, filter_size)
                lh_map_loss = self.l2loss(lh_pred_map, label_maps[:, chosen_idxes])
                lh_bbox_loss = self.smoothl1loss(lh_pred_bbox, gt_bboxes[:, chosen_idxes], anchors)
            meta_loss_update = lh_map_loss + lh_bbox_loss

            map_loss_total += lh_map_loss
            bbox_loss_total += lh_bbox_loss
            meta_loss_total += meta_loss_update

        map_loss_avg = map_loss_total / config.time_step
        bbox_loss_avg = bbox_loss_total / config.time_step
        meta_loss_avg = meta_loss_total / config.time_step
        return map_loss_avg, bbox_loss_avg, meta_loss_avg


class OnlineROAM(nn.Module):

    def __init__(self, feat_extractor):
        super(OnlineROAM, self).__init__()
        self.feat_extractor = feat_extractor
        self.meta_init = MetaInitializer()
        self.meta_opti = MetaOptimizer()
        self.l2loss = L2normLoss()
        self.smoothl1loss = SmoothL1Loss()
        self.meta_opti.eval()

    def compute_resize_feature(self, patch, win_cell_sz):
        with torch.no_grad():
            feat = self.feat_extractor(patch)
            feat = F.interpolate(feat, (win_cell_sz, win_cell_sz), mode='bilinear', align_corners=True)
        return feat

    def initialize(self, feat, label_map, gt_bbox):

        # build initial target classifier
        init_params, init_lrs = self.meta_init.initialize()
        self.meta_opti.initialize(init_lrs)
        target_cell_sz = torch.ceil(gt_bbox[:, 0, 2:]/config.cell_sz)
        filter_size = compute_filter_size(target_cell_sz)
        base_anchor_sizes = filter_size/config.filter_scale*config.cell_sz
        self.map_size = feat.shape[3]
        self.anchors = generate_anchors(self.map_size, base_anchor_sizes)

        # calculate initial update loss
        pred_map, pred_bbox = adaptable_conv2d(feat, init_params, filter_size)
        map_loss = self.l2loss(pred_map, label_map)
        bbox_loss = self.smoothl1loss(pred_bbox, gt_bbox, self.anchors)
        loss = map_loss + bbox_loss
        # meta update
        grads = torch.autograd.grad(loss, init_params)
        self.updated_params = init_params - init_lrs * grads[0]
        pred_map, pred_bbox = adaptable_conv2d(feat, self.updated_params, filter_size)
        lh_map_loss = self.l2loss(pred_map, label_map)
        lh_bbox_loss = self.smoothl1loss(pred_bbox, gt_bbox, self.anchors)
        lh_loss = lh_map_loss + lh_bbox_loss
        print(map_loss.data.item(), lh_map_loss.data.item())
        print(bbox_loss.data.item(), lh_bbox_loss.data.item())
        print(loss.data.item(), lh_loss.data.item())

        self.updating_feats = []
        self.updating_maps = []
        self.updating_bboxes = []
        self.filter_size = filter_size
        self.ref_score = np.max(pred_map.data.cpu().numpy())

    def save_training_samples(self, updating_feat, updating_map, updating_bbox):

        if len(self.updating_feats) == config.max_db_size:
            self.updating_feats.pop(0)
            self.updating_maps.pop(0)
            self.updating_bboxes.pop(0)
        self.updating_feats.append(updating_feat)
        self.updating_maps.append(updating_map)
        self.updating_bboxes.append(updating_bbox)

    def adapt_filter_size(self, target_sz):

        # adapt filter to new size
        target_sz = torch.from_numpy(target_sz[None]).float()
        if torch.cuda.is_available():
            target_sz = target_sz.cuda()
        filter_size = compute_filter_size(target_sz)
        if (self.filter_size != filter_size).any():
            self.filter_size = filter_size
            base_anchor_sizes = self.filter_size / config.filter_scale * config.cell_sz
            # self.anchors = generate_anchors(self.map_size, base_anchor_sizes)
            self.anchors[:, 2:] = base_anchor_sizes

    def meta_update(self):

        # update model
        updating_feats = torch.stack(self.updating_feats, 1)
        updating_maps = torch.stack(self.updating_maps, 1)
        updating_bboxes = torch.stack(self.updating_bboxes, 1)
        n_total = config.n_online_updates * config.n_update_batch
        n_cur_db = updating_feats.size(1)
        all_indices = [i%n_cur_db for i in range(n_total)]
        # all_indices = np.random.permutation(all_indices)
        indices = [all_indices[i*config.n_update_batch:(i+1)*config.n_update_batch]
                   for i in range(config.n_online_updates)]

        for i in range(config.n_online_updates):
            pred_map, pred_bbox = adaptable_conv2d(updating_feats[:, indices[i]], self.updated_params, self.filter_size)
            map_loss = self.l2loss(pred_map, updating_maps[:, indices[i]])
            bbox_loss = self.smoothl1loss(pred_bbox, updating_bboxes[:, indices[i]], self.anchors)
            loss = map_loss + bbox_loss
            grads = torch.autograd.grad(loss, self.updated_params)
            self.updated_params = self.meta_opti.meta_update(self.updated_params, loss, grads[0])

    def localize(self, feat):
        pred_map, pred_bbox = adaptable_conv2d(feat, self.updated_params, self.filter_size)
        return pred_map, pred_bbox