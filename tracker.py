# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import config
import numpy as np
import torch
import torchvision.transforms as trans
from utils import get_search_size, get_search_patch, gaussian_shaped_labels, \
    valid_bbox, InterResultShow, un_preprocess_image, bbox_transform_inv
from roam import OnlineROAM
import cv2

class Tracker():

    def __init__(self, feat_extractor, is_debug=False):

        self.roam = OnlineROAM(feat_extractor)
        if torch.cuda.is_available():
            self.roam = self.roam.cuda()

        self.preprocessor = trans.Compose(
            [trans.ToTensor(),
             trans.Normalize(mean=config.mean, std=config.std)])
        self.pad_value = np.array(config.mean)

        self.is_debug = is_debug

    def load_models(self, model_file):

        print('Loaded {} for tracking'.format(model_file))
        roam_state = torch.load(model_file) if torch.cuda.is_available() \
            else torch.load(model_file, map_location=lambda storage, loc: storage)
        self.roam.load_state_dict(roam_state, strict=False)

    def initialize(self, init_image, init_bbox, seq_name=None):

        self.seq_name = seq_name
        self.max_obj_sz = config.base_target_sz
        init_bbox = np.array(init_bbox)

        # calculate the scale between base window and actual window
        base_target_sz = np.array([config.base_target_sz, config.base_target_sz])
        base_window_sz = get_search_size(base_target_sz, config.search_scale)
        base_window_cell_sz = np.ceil(base_window_sz / config.cell_sz)
        base_window_cell_sz = base_window_cell_sz - (base_window_cell_sz % 2) + 1

        init_patches, label_maps, gt_bboxes = [], [], []
        for ratio in config.aug_init_ratios:
            for scale in config.aug_init_scales:
                height, width = init_image.shape[0: 2]
                sw, sh = int(width * ratio), int(height / ratio)
                image_resized = cv2.resize(init_image, (sw, sh))
                bbox_reiszed = init_bbox * np.array([ratio, 1 / ratio, ratio, 1 / ratio])
                search_scale = config.search_scale / scale
                window_sz = get_search_size(bbox_reiszed[2:], search_scale)

                # crop image patches
                target_center = bbox_reiszed[:2] + np.floor(bbox_reiszed[2:] / 2)
                init_patch = get_search_patch(image_resized, target_center, self.pad_value, window_sz, base_window_sz)
                init_patch = self.preprocessor(init_patch).unsqueeze(0)
                init_patches.append(init_patch)

                # calculate labeled response map for meta training
                scale = base_window_sz[0]/window_sz[0]
                scaled_bbox = init_bbox*scale
                target_cell_sz = np.ceil(scaled_bbox[2:] / config.cell_sz)
                output_sigma = target_cell_sz * config.output_sigma_factor
                label_map = gaussian_shaped_labels(output_sigma, base_window_cell_sz)
                label_map = torch.from_numpy(label_map[None, None, :]).float()
                label_maps.append(label_map)

                # calculate bbox on roi patch
                bbox_center = np.zeros(2, np.float32)
                bbox_on_patch = np.concatenate([bbox_center, scaled_bbox[2:]])
                gt_bbox = torch.from_numpy(bbox_on_patch[None, :]).float()
                gt_bboxes.append(gt_bbox)

        init_patches = torch.stack(init_patches, 1)
        label_maps = torch.stack(label_maps, 1)
        gt_bboxes = torch.stack(gt_bboxes, 1)
        if torch.cuda.is_available():
            init_patches = init_patches.cuda()
            label_maps = label_maps.cuda()
            gt_bboxes = gt_bboxes.cuda()

        # calculate motion map to constraint large movement
        window_sz = get_search_size(init_bbox[2:], config.search_scale)
        scale = base_window_sz[0] / window_sz[0]
        scaled_bbox = init_bbox * scale
        target_cell_sz = np.ceil(scaled_bbox[2:] / config.cell_sz)
        motion_sigma = target_cell_sz * config.motion_sigma_factor
        motion_map = gaussian_shaped_labels(motion_sigma, base_window_cell_sz)
        motion_map = motion_map.flatten().astype(np.float32)
        motion_sigma_loose = target_cell_sz * config.motion_sigma_factor*2
        motion_map_loose = gaussian_shaped_labels(motion_sigma_loose, base_window_cell_sz)
        motion_map_loose = motion_map_loose.flatten().astype(np.float32)

        # meta-learn an initial tracker using initial patch and corresponding labeled map
        init_feats = self.roam.compute_resize_feature(init_patches[0], int(base_window_cell_sz[0]))
        self.roam.initialize(init_feats[None, :], label_maps, gt_bboxes)

        # set target center and target size
        self.target_center = init_bbox[:2] + np.floor(init_bbox[2:] / 2)
        self.target_sz = init_bbox[2:]

        # save some unchanged parameters for subsequent tracking
        self.base_window_sz = base_window_sz
        self.base_window_cell_sz = base_window_cell_sz
        self.motion_map = motion_map
        self.motion_map_loose = motion_map_loose
        self.loose_motion_penalty = True
        np.random.seed(123)
        torch.manual_seed(456)
        torch.cuda.manual_seed(789)

        self.frame_idx = 0

        if self.is_debug:
            self.patches = []
            self.inter_result_show = InterResultShow(init_image, init_bbox, 0,
                                                     init_patches[0, 0].data.cpu().numpy(),
                                                     label_maps[0, 0, 0].cpu().numpy())
        return init_bbox

    def track(self, image):
        self.frame_idx += 1

        window_sz = get_search_size(self.target_sz, config.search_scale)

        # -------------- bounding box estimation --------------------------------
        patch = get_search_patch(image, self.target_center, self.pad_value, window_sz, self.base_window_sz)
        patch = self.preprocessor(patch).unsqueeze(0)
        if torch.cuda.is_available():
            patch = patch.cuda()
        feat = self.roam.compute_resize_feature(patch, int(self.base_window_cell_sz[0]))
        response, delta_bbox = self.roam.localize(feat[:, None])
        response = response.data.cpu().numpy()
        delta_bbox = delta_bbox.view(-1, 4)
        delta_bbox[torch.isnan(delta_bbox)]=0
        pred_bboxes = bbox_transform_inv(self.roam.anchors, delta_bbox)
        pred_bboxes = pred_bboxes.data.cpu().numpy()

        def change(r):
            return np.maximum(r, 1./r)

        scale = self.base_window_sz[0]/window_sz[0]
        scale_change = change(np.sqrt(np.prod(pred_bboxes[:, 2:], 1)) / np.sqrt(np.prod(self.target_sz*scale)))
        ratio_change = change((pred_bboxes[:, 2]/pred_bboxes[:, 3]) / (self.target_sz[0]/self.target_sz[1]))
        score_penalty = np.exp(-(scale_change*ratio_change-1.)*config.score_penalty_factor)
        response_p = response.flatten()*score_penalty
        response_pp = response_p * self.motion_map.flatten()
        max_score = np.max(response_pp)
        # print('{}/{}'.format(max_score, self.roam.ref_score))
        # handle drift case
        if config.handle_drift:
            if max_score < config.miss_thres * self.roam.ref_score or self.loose_motion_penalty:
                response_pp = response_p * self.motion_map_loose
                max_score = np.max(response_pp)
                self.loose_motion_penalty = False \
                    if max_score > config.reliable_thres * self.roam.ref_score else True
                # print('new_max_score: {}'.format(max_score))
        max_idx = np.argmax(response_pp)
        pred_bbox = pred_bboxes[max_idx]

        self.target_center += pred_bbox[0:2] / scale
        size_decay = score_penalty[max_idx] * response_p[max_idx]* config.size_decay
        self.target_sz = (1 - size_decay) * self.target_sz + size_decay * pred_bbox[2:] / scale
        self.target_center, self.target_sz = valid_bbox(self.target_center, self.target_sz, image.shape)

        # ------------- output bounding box --------------------------------
        target_bbox = np.zeros(4)
        target_bbox[:2] = self.target_center - self.target_sz / 2
        target_bbox[2:] = self.target_sz

        # ------------------ tracking model updating ------------------------

        pred_bbox[2:] = self.target_sz*scale
        target_cell_sz = np.ceil(pred_bbox[2:] / config.cell_sz)
        output_sigma = target_cell_sz * config.output_sigma_factor
        map_center = np.round(pred_bbox[:2]/ config.cell_sz) + np.floor(self.base_window_cell_sz/2)
        label_map = gaussian_shaped_labels(output_sigma, self.base_window_cell_sz, map_center)
        label_map = torch.from_numpy(label_map[None, :]).float()
        pred_bbox = torch.from_numpy(pred_bbox).float()
        if torch.cuda.is_available():
            label_map = label_map.cuda()
            pred_bbox = pred_bbox.cuda()
        if config.handle_drift:
            if max_score > config.miss_thres * self.roam.ref_score:
                self.roam.save_training_samples(feat, label_map[None, :], pred_bbox[None, :])
        else:
            self.roam.save_training_samples(feat, label_map[None, :], pred_bbox[None, :])

        if self.frame_idx % config.update_interval == 0:
            self.roam.adapt_filter_size(self.target_sz)
            self.roam.meta_update()

        # ------------------ debug show -------------------------------------
        if self.is_debug:
            # show responses and bbox on search image
            patch_np = patch[0].data.cpu().numpy().copy()
            patch_np = un_preprocess_image(patch_np)
            response = response.reshape(self.base_window_cell_sz.astype(np.int).tolist())
            self.inter_result_show.display_inter_result(self.frame_idx, image,
                                                        patch_np,
                                                        response.copy())
            self.inter_result_show.display_bbox(target_bbox, self.seq_name, False)
            if len(self.patches) == config.max_db_size:
                self.patches.pop(0)
            self.patches.append(patch_np)
            if self.frame_idx % config.update_interval == 0:
                self.inter_result_show.display_examples(self.patches)

        return target_bbox


