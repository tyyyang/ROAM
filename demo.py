# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import config
import os
from utils import list_models
from networks import FeatureExtractor
from tracker import Tracker
import time
from utils import compute_success_overlap, get_axis_aligned_bbox
import numpy as np
from PIL import Image


def load_seq_config(data_root, seq_name):

    src = os.path.join(data_root, seq_name, 'groundtruth_rect.txt')
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    img_path = os.path.join(data_root, seq_name, 'img')
    img_names = sorted(os.listdir(img_path))
    frame_paths = [os.path.join(img_path, img_name) for img_name in img_names]

    return gt_rects, frame_paths

def OTB_run(gt_rects, frame_paths, tracker):

    tic = time.time()
    # tracking loop
    res = []
    for idx, img_path in enumerate(frame_paths):
        print('Frame', idx)
        img = np.array(Image.open(frame_paths[idx]).convert('RGB'))
        if idx == 0:
            if len(gt_rects[0]) == 8:
                init_bbox = get_axis_aligned_bbox(np.array(gt_rects[0]))
            else:
                init_bbox = gt_rects[0]
            pred_bbox = tracker.initialize(img, init_bbox)
        else:
            pred_bbox = tracker.track(img)
        res.append(pred_bbox)
    fps = len(res) / (time.time() - tic)
    success_overlap = compute_success_overlap(gt_rects, res)
    print('success overlap: %.4f, fps:%.2f' % (success_overlap.mean(), fps))


if __name__ == '__main__':

    gt_rects, frame_paths = load_seq_config(config.otb_dir, 'Trans')
    feat_extractor = FeatureExtractor(config.feat_dir)
    tracker = Tracker(feat_extractor, is_debug=True)
    models = list_models(os.path.abspath(config.model_dir))
    tracker.load_models(models[-1])

    OTB_run(gt_rects, frame_paths, tracker)