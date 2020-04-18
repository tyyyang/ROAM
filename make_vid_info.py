# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import config

max_trackid = 50
min_seq_len = 50

class BoundingBox(object):
    pass

def get_item(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1

def get_int(name, root, index=0):
    return int(get_item(name, root, index))

def find_num_bb(root):
    index = 0
    while True:
        if get_int('xmin', root, index) == -1:
            break
        index += 1
    return index

def process_xml(xml_file):
    """Process a single XML file containing a bounding box."""
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print('Failed to parse: ' + xml_file, file=sys.stderr)
        return None

    root = tree.getroot()
    num_boxes = find_num_bb(root)
    boxes = []
    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = get_int('xmin', root, index)
        box.ymin = get_int('ymin', root, index)
        box.xmax = get_int('xmax', root, index)
        box.ymax = get_int('ymax', root, index)
        box.trackid = get_int('trackid', root, index)

        file_name = get_item('filename', root) + '.JPEG'
        folder = get_item('folder', root)

        box.width = get_int('width', root)
        box.height = get_int('height', root)
        box.img_path = os.path.join(folder, file_name)
        box.label = get_item('name', root)
        box.visible = (get_item('occluded', root)!='0')
        boxes.append(box)

    return boxes

def check_ratio(frame):

    bb_width = frame.xmax - frame.xmin + 1
    bb_height = frame.ymax -frame.ymin + 1
    ratio1 = bb_height/bb_width
    ratio2 = bb_width/bb_height
    ratio = min(ratio1, ratio2)
    return ratio > 0.1

def make_seqs_by_anno(seq_anno_dir):

    files = sorted(os.listdir(seq_anno_dir))
    frames = []
    for file in files:
        bboxes = process_xml(os.path.join(seq_anno_dir, file))
        id_bboxes = max_trackid * [None]
        for bbox in bboxes:
            id = bbox.trackid
            if id >= max_trackid:
                print(bbox.img_path)
            id_bboxes[id] = bbox
        frames.append(id_bboxes)
    # traverse all ids to construct several seqs
    seqs = []
    for id in range(max_trackid):
        images = []
        for frame in frames:
            if frame[id] is not None:
                images.append(frame[id])
        if len(images) >= min_seq_len:
            seqs.append(images)
    return seqs

def convert_to_dict(seqs, is_valid):
    seqs_new = []
    n_generated_clips, n_removed_clips = 0, 0
    for seq in seqs:
        seq_info = {}
        img_path = seq[0].img_path
        strs = img_path.split('/')
        if is_valid:
            seq_name = strs[0]
            start_frame = int(strs[1][:-5])+1
            end_frame = start_frame + len(seq) - 1
        else:
            seq_name = strs[0]+'/'+strs[1]
            start_frame = int(strs[2][:-5])+1
            end_frame = start_frame + len(seq) - 1
        gt, visible = [], []
        for frame in seq:
            bbox = [frame.xmin, frame.ymin,
                    frame.xmax - frame.xmin + 1,
                    frame.ymax - frame.ymin + 1]
            gt.append(bbox)
            # visible.append(frame.visible)
        # img_width = seq[0].width
        # img_height = seq[0].height
        gt_numpy = np.array(gt)
        visible = (gt_numpy[:, 2] > 0) & (gt_numpy[:, 3] > 0)
        visible = visible.tolist()
        n_visible = np.count_nonzero(visible)
        if n_visible < min_seq_len:
            print('{} does not enough visible objects {}/{}'
                  .format(seq_name, n_visible, len(visible)))
            n_removed_clips += 1
            continue

        n_generated_clips += 1
        seq_info['seq_name'] = seq_name
        seq_info['start_frame'] = start_frame
        seq_info['end_frame'] = end_frame
        seq_info['gt_bboxes'] = gt
        # seq_info['im_width'] = img_width
        # seq_info['im_height'] = img_height
        seq_info['visible'] = visible
        seqs_new.append(seq_info)

    return seqs_new, n_removed_clips, n_generated_clips


def make_dataset(root_dir, is_train):

    seqs = []
    n_generated_clips, n_removed_clips = 0, 0
    if is_train:
        anno_dir = os.path.join(root_dir, 'Annotations/VID/train')
        dirs1 = sorted(os.listdir(anno_dir))
        for dir1 in dirs1:
            dirs2 = sorted(os.listdir(os.path.join(anno_dir, dir1)))
            for dir2 in dirs2:
                seq_anno_dir = os.path.join(anno_dir, dir1, dir2)
                seq = make_seqs_by_anno(seq_anno_dir)
                seq_info, n_removed, n_generated = convert_to_dict(seq, False)
                seqs += seq_info
                n_removed_clips += n_removed
                n_generated_clips += n_generated

        data_dir = os.path.join(root_dir, 'Data/VID/train')
        json.dump(seqs, open(os.path.join(data_dir, 'train.json'), 'w'))
        print('Finish training data with {}/{} video clips removed'.format(n_removed_clips, n_generated_clips))
    else:
        anno_dir = os.path.join(root_dir, 'Annotations/VID/val')
        dirs1 = sorted(os.listdir(anno_dir))
        for dir1 in dirs1:
            seq_anno_dir = os.path.join(anno_dir, dir1)
            seq = make_seqs_by_anno(seq_anno_dir)
            seq_info, n_removed, n_generated = convert_to_dict(seq, True)
            seqs += seq_info
            n_removed_clips += n_removed
            n_generated_clips += n_generated

        # seqs = crop_images(seqs, data_dir)
        # pickle.dump(seqs, open(os.path.join(root_dir, 'val.pk'), 'wb'))
        data_dir = os.path.join(root_dir, 'Data/VID/val')
        json.dump(seqs, open(os.path.join(data_dir, 'val.json'), 'w'))
        print('Finish validation data with {}/{} video clips removed'.format(n_removed_clips, n_generated_clips))

if __name__ == '__main__':

    root_dir = config.root_dir + '/Data/ILSVRC'
    make_dataset(root_dir, True)
    make_dataset(root_dir, False)
