# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import config
import numpy as np
import os
import torch
import torchvision.transforms as trans
import json
from torch.utils.data import Dataset
from utils import get_search_patch, get_search_size, gaussian_shaped_labels, default_loader
import cv2

class BaseDataset(Dataset):

    def __init__(self, train_dir, val_dir=None, is_train=True):
        if is_train:
            self._seq_dir = train_dir
            with open(os.path.join(train_dir, 'train.json'), 'r') as f:
                self._seqs = json.load(f)
        else:
            np.random.seed(123)
            self._seq_dir = val_dir
            with open(os.path.join(val_dir, 'val.json'), 'r') as f:
                self._seqs = json.load(f)
        self.n_seq = len(self._seqs)
        print(self.n_seq)
        self.is_train = is_train
        self.seq_id = -1

        self.preprocessor = trans.Compose(
            [trans.ToTensor(),
             trans.Normalize(mean=config.mean, std=config.std)])
        self.pad_value = np.array(config.mean)

        base_target_sz = np.array([config.base_target_sz, config.base_target_sz])
        self.base_window_sz = get_search_size(base_target_sz, config.search_scale)
        window_cell_sz = np.ceil(self.base_window_sz / config.cell_sz)
        self.window_cell_sz = window_cell_sz - (window_cell_sz % 2) + 1


    def __len__(self):
        return len(self._seqs)

    def __getitem__(self, seq_id):

        # uniformly choosing video frames
        seq_name = self._seqs[seq_id]['seq_name']
        start_frame = self._seqs[seq_id]['start_frame']
        end_frame = self._seqs[seq_id]['end_frame']
        bboxes = self._seqs[seq_id]['gt_bboxes']
        visible = self._seqs[seq_id]['visible']
        visible_frame_idxes = np.arange(0, end_frame - start_frame + 1)[visible]
        seq_len = len(visible_frame_idxes)
        clip_len = config.look_ahead * config.time_step + 1
        assert seq_len >= clip_len
        start_idx = np.random.randint(seq_len - clip_len + 1) if self.is_train else 0
        selected_idxes = [visible_frame_idxes[idx] for idx in range(start_idx, start_idx + clip_len)]

        # build training examples for initial frame
        patches, label_maps, gt_bboxes = [], [], []
        img_path = self._get_img_path(seq_name, start_frame, selected_idxes[0])
        init_image = default_loader(img_path)
        init_bbox = np.array(bboxes[selected_idxes[0]])
        for ratio in config.aug_init_ratios:
            for scale in config.aug_init_scales:
                # aspect ratio augmentation
                height, width = init_image.shape[0: 2]
                sw, sh = int(width * ratio), int(height / ratio)
                image_resized = cv2.resize(init_image, (sw, sh))
                bbox_reiszed = init_bbox * np.array([ratio, 1 / ratio, ratio, 1 / ratio])
                # scale changes augmentation
                search_scale = config.search_scale / scale
                # generate training examples
                patch, label_map, bbox_on_patch = self._generate_training_examples(image_resized, bbox_reiszed, search_scale)
                patches.append(patch)
                label_maps.append(label_map)
                gt_bboxes.append(bbox_on_patch)

        # build training examples for subsequent frames.
        for i, idx in enumerate(selected_idxes[1:]):
            img_path = self._get_img_path(seq_name, start_frame, idx)
            image = default_loader(img_path)
            bbox = np.array(bboxes[idx])
            # aspect ratio augmentation
            height, width = image.shape[0: 2]
            ratio = np.random.uniform(config.aug_ratios_range[0], config.aug_ratios_range[1])
            sw, sh = int(width * ratio), int(height / ratio)
            image = cv2.resize(image, (sw, sh))
            bbox = bbox * np.array([ratio, 1 / ratio, ratio, 1 / ratio])
            # scale changes augmentation
            obj_scale = np.random.uniform(config.aug_scales_range[0], config.aug_scales_range[1])
            search_scale = config.search_scale/obj_scale
            # generate training examples
            patch, label_map, bbox_on_patch = self._generate_training_examples(image, bbox, search_scale)
            patches.append(patch)
            label_maps.append(label_map)
            gt_bboxes.append(bbox_on_patch)

        patches = torch.stack(patches, 0)
        label_maps = torch.stack(label_maps, 0)
        gt_bboxes = torch.stack(gt_bboxes, 0)

        return patches, label_maps, gt_bboxes

    def _get_img_path(self, seq_name, start_frame, index):
        raise NotImplementedError

    def _generate_training_examples(self, image, bbox, search_scale):
        # get roi patches
        window_sz = get_search_size(bbox[2:], search_scale)
        scale = self.base_window_sz[0] / window_sz[0]
        offset = np.random.randint(-config.offset_range, config.offset_range, 2)
        center = bbox[:2] + np.floor(bbox[2:] / 2) - offset/scale
        patch = get_search_patch(image, center, self.pad_value, window_sz, self.base_window_sz)
        patch = self.preprocessor(patch)

        # get label maps
        scaled_bbox = np.round(bbox * scale)
        target_cell_sz = np.ceil(scaled_bbox[2:] / config.cell_sz)
        output_sigma = target_cell_sz * config.output_sigma_factor
        map_center = np.floor(self.window_cell_sz / 2) + offset //config.cell_sz
        label_map = gaussian_shaped_labels(output_sigma, self.window_cell_sz, map_center)
        label_map = torch.from_numpy(label_map[None, :]).float()

        # get gt bboxes
        bbox_center = np.zeros(2, np.float32) + offset
        bbox_on_patch = np.concatenate([bbox_center, scaled_bbox[2:]])
        bbox_on_patch = torch.from_numpy(bbox_on_patch).float()
        return patch, label_map, bbox_on_patch


class VIDataset(BaseDataset):

    def __init__(self, root_dir, is_train):
        train_dir = os.path.join(root_dir, 'Data/ILSVRC/Data/VID/train')
        val_dir = os.path.join(root_dir, 'Data/ILSVRC/Data/VID/val')
        super(VIDataset, self).__init__(train_dir, val_dir, is_train)

    def _get_img_path(self, seq_name, start_frame, index):
        img_path = self._seq_dir + ('/%s/%06d.JPEG' % (seq_name, (start_frame - 1) + index))
        return img_path