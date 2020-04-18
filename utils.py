# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

from torch.nn import functional as F
import torch
import numpy as np
import cv2
import math
import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
import config
import glob

# import matplotlib
# matplotlib.rcParams['backend'] = "Qt5Agg"

# data input utils --------------------------------------------------------

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def get_search_size(target_sz, scale):

    window_sz = np.sqrt(np.prod(target_sz)) * scale
    # workround for strange nan error for lasot dataset
    if np.isnan(window_sz):
        window_sz = int(config.base_target_sz*scale)
    else:
        window_sz = int(window_sz)
    window_sz = int(window_sz - (window_sz % 2) + 1)
    return np.array([window_sz, window_sz])

def crop_image(img, center, sz, pad_value):

    # center_x, center_y = np.array(center, dtype='float32') - 1
    center_x,center_y = np.array(center,dtype='float32')
    w, h = np.array(sz, dtype='float32')
    half_w, half_h = w / 2, h / 2

    img_h, img_w, _ = img.shape
    min_x = np.floor(center_x - half_w + 0.5).astype(int)
    min_y = np.floor(center_y - half_h + 0.5).astype(int)
    max_x = np.floor(center_x + half_w + 0.5).astype(int)
    max_y = np.floor(center_y + half_h + 0.5).astype(int)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]
    else:
        max_x_val = max(1, min(img_w, max_x))
        max_y_val = max(1, min(img_h, max_y))
        min_x_val = min(max(0, min_x), max_x_val - 1)
        min_y_val = min(max(0, min_y), max_y_val - 1)
        # print(min_x_val, min_y_val, max_x_val, max_y_val)
        cropped = []
        for i in range(3):
            cropped.append(pad_value[i]*np.ones((max_y - min_y, max_x - min_x), dtype=np.float32))
        cropped = np.stack(cropped, 2)
        # cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='float32')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    return cropped

def get_search_patch(img, target_center, pad_value, window_sz, base_window_sz):

    img_array = np.asarray(img, dtype='float32')

    if window_sz[0] > max(img.shape):
        scale = base_window_sz[0] / window_sz[0]
        img_array = cv2.resize(img_array,
                                  (int(round(img.shape[1]*scale)), int(round(img.shape[0]*scale))),
                                  cv2.INTER_AREA)
        patch = crop_image(img_array, target_center*scale, base_window_sz, pad_value)
    else:
        patch = crop_image(img_array, target_center, window_sz, pad_value)
        patch = cv2.resize(patch, tuple(base_window_sz), cv2.INTER_AREA)
    return patch

def gaussian_shaped_labels(sigma, win_sz, center=None):

    if center is None:
        center = np.floor(win_sz / 2)

    v = np.linspace(0, win_sz[0]-1, win_sz[0]) - center[0]
    u = np.linspace(0, win_sz[1]-1, win_sz[1]) - center[1]
    xs, ys = np.meshgrid(v, u)

    labels = np.exp(-config.alpha * ((xs ** 2) / (sigma[0] ** 2) + (ys ** 2) / (sigma[1] ** 2)))
    return labels

def un_preprocess_image(img):
    img = img.transpose(1, 2, 0)
    for i in range(3):
        img[:, :, i] = img[:, :, i] * config.std[i] + config.mean[i]

    img = img.astype(np.uint8)
    return img


# training meta trackers utils ---------------------------------------------
def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1.view(-1,1), x2.view(-1,1)), 1)

def compute_filter_size(target_szs):

    min_values = torch.ones_like(target_szs)
    if torch.cuda.is_available():
        min_values = min_values.cuda()
    target_szs = torch.max(target_szs, min_values)
    aspect_ratio = target_szs[:, 0]/target_szs[:, 1]
    cfactor = torch.sqrt(config.base_target_sz**2 / aspect_ratio)
    norm_target_szs = torch.stack([aspect_ratio*cfactor, cfactor], 1)
    filter_sizes = torch.ceil(norm_target_szs / config.cell_sz * config.filter_scale)
    filter_sizes = filter_sizes - (filter_sizes % 2) + 1
    return filter_sizes

def adaptable_conv2d(feats, params, filter_sizes):

    cf_conv_weight, cf_conv_bias, cf_weight, cf_bias, offset = restore_flatparam(params, 1, 0)
    reg_conv_weight, reg_conv_bias, reg_weight, reg_bias, offset = restore_flatparam(params, 4, offset)
    feat_shape = feats.shape
    assert len(feat_shape) == 5
    feats = feats.contiguous().view([-1]+list(feats.shape[2:]))
    cf_conv_feats = F.conv2d(feats, cf_conv_weight, cf_conv_bias)
    cf_conv_feats = cf_conv_feats.view(list(feat_shape[:2]) + list(cf_conv_feats.shape[1:]))

    reg_conv_feats = F.conv2d(feats, reg_conv_weight, reg_conv_bias)
    reg_conv_feats = reg_conv_feats.view(list(feat_shape[:2]) + list(reg_conv_feats.shape[1:]))

    pred_map, pred_bbox = [], []
    for i, (cf_conv_feat, reg_conv_feat, fs) in enumerate(zip(cf_conv_feats, reg_conv_feats, filter_sizes)):
        adapt_cf_weight = F.interpolate(cf_weight, (int(fs[1]), int(fs[0])), mode='bilinear', align_corners=True)
        output = F.conv2d(cf_conv_feat, adapt_cf_weight, cf_bias, padding=(int(fs[1]//2), int(fs[0]//2)))
        pred_map.append(output)
        adapt_reg_weight = F.interpolate(reg_weight, (int(fs[1]), int(fs[0])), mode='bilinear', align_corners=True)
        output = F.conv2d(reg_conv_feat, adapt_reg_weight, reg_bias, padding=(int(fs[1] // 2), int(fs[0] // 2)))
        pred_bbox.append(output.permute(0, 2, 3, 1))
    pred_map = torch.stack(pred_map, 0)
    pred_bbox = torch.stack(pred_bbox, 0)
    return pred_map, pred_bbox

def restore_flatparam(params, out_channels, offset):

    conv_weight = params[offset: offset + config.feat_channels * config.cf_channels].view(config.cf_channels, config.feat_channels, 1, 1)
    offset += config.feat_channels * config.cf_channels
    conv_bias = params[offset: offset + config.cf_channels]
    offset += config.cf_channels
    weigh_size = config.base_filter_size[0] * config.base_filter_size[1] * config.cf_channels * out_channels
    cf_weight = params[offset:offset + weigh_size].view(out_channels, config.cf_channels, config.base_filter_size[1],
                                                        config.base_filter_size[0])
    offset += weigh_size
    cf_bias = params[offset: offset + out_channels]
    offset += out_channels
    return conv_weight, conv_bias, cf_weight, cf_bias, offset

def bbox_transform(anchors, gt_bboxes):

    an_ctr_x, an_ctr_y, an_widths, an_heights = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = gt_bboxes[:, 0], gt_bboxes[:, 1], gt_bboxes[:, 2], gt_bboxes[:, 3]

    dx = (gt_ctr_x - an_ctr_x) / an_widths
    dy = (gt_ctr_y - an_ctr_y) / an_heights
    dw = torch.log(gt_widths / an_widths)
    dh = torch.log(gt_heights / an_heights)

    return torch.stack([dx, dy, dw, dh], 1)

def bbox_transform_inv(anchors, pred_deltas):

    ctr_x, ctr_y, widths, heights = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    dx, dy, dw, dh = pred_deltas[:, 0], pred_deltas[:, 1], pred_deltas[:, 2], pred_deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    return torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h], 1)

def generate_anchors(feat_size, base_sizes):

    base_anchors = torch.zeros([base_sizes.shape[0], 4])
    base_anchors[:, 2:] = base_sizes
    feat_height = feat_width = feat_size
    half_width, half_height = feat_width//2, feat_height//2
    sx = torch.arange(-half_width, feat_width-half_width, dtype=torch.float32) * config.cell_sz
    sy = torch.arange(-half_height, feat_height-half_height, dtype=torch.float32) * config.cell_sz
    shift_y, shift_x = torch.meshgrid([sx, sy])
    shifts = torch.stack([shift_x.contiguous().view(-1), shift_y.contiguous().view(-1)], 1)

    A = base_anchors.shape[0]
    K = shifts.shape[0]

    all_anchors = base_anchors.repeat(1, K).view(-1, 4)
    shifts = shifts.repeat(A, 1)
    all_anchors[:, :2] += shifts
    if torch.cuda.is_available():
        all_anchors = all_anchors.cuda()
    return all_anchors

def generate_delta_bboxes(anchors, gt_bboxes):

    bs = gt_bboxes.shape[0]
    ts = gt_bboxes.shape[1]
    anchors = anchors.view(bs, 1, -1, 4).repeat(1, ts, 1, 1)
    gt_bboxes = gt_bboxes.view(bs, ts, 1, 4).repeat(1, 1, anchors.shape[2], 1)
    anchors = anchors.view(-1, 4)
    gt_bboxes = gt_bboxes.view(-1, 4)
    anchors_trans = torch.cat([anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, 2:]], 1)
    gt_bbox_trans = torch.cat([gt_bboxes[:, :2] - gt_bboxes[:, 2:] / 2, gt_bboxes[:, 2:]], 1)
    # if len(gt_bbox_trans) < 4:
    #     print(gt_bbox_trans, gt_bbox)
    iou = compute_iou(anchors_trans, gt_bbox_trans)
    mask = torch.zeros([len(anchors), 4], dtype=torch.bool)
    if torch.cuda.is_available():
        mask = mask.cuda()
    # pos_idxes = torch.where(iou > config.anchor_pos_thres)[0]
    # pos_idxes = torch.argsort(iou)[-config.anchor_pos_num*ts:]
    sorted_iou, sorted_idxes = torch.sort(iou)
    pos_idxes = sorted_idxes[-config.anchor_pos_num*ts:]
    # tmp = anchors[pos_idxes]
    mask[pos_idxes, :] = 1

    delta_bboxes = bbox_transform(anchors, gt_bboxes)
    return delta_bboxes, mask

def compute_iou(rect1, rect2):

    if len(rect1.shape)==1:
        rect1 = rect1[None,:]
    if len(rect2.shape)==1:
        rect2 = rect2[None,:]

    left = torch.max(rect1[:,0], rect2[:,0])
    right = torch.min(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = torch.max(rect1[:,1], rect2[:,1])
    bottom = torch.min(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    tensor_zero = torch.tensor(0, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor_zero = tensor_zero.cuda()
    intersect = torch.max(tensor_zero, right - left) * torch.max(tensor_zero, bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = torch.clamp(intersect / union, 0, 1)
    return iou


# tracking utils------------------------------------------------------------
def list_models(model_dir=None):
    model_pattern = 'roam_epoch*.torch'
    if model_dir is None:
        model_dir = config.model_dir
    model_files = sorted(glob.glob(os.path.join(model_dir, model_pattern)))
    return model_files

def valid_bbox(pos, sz, img_shape):
    img_sz = np.array([img_shape[1], img_shape[0]])
    pos = np.minimum(np.maximum(pos, 0), img_sz)
    sz = np.maximum(np.minimum(sz, img_sz), 10)
    return pos, sz

def overlap_ratio(rect1, rect2):

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def compute_success_overlap(gt_bb, result_bb):

    thresholds_overlap = np.arange(0, 1.05, 0.05)
    n_frame = len(gt_bb)
    success = np.zeros(len(thresholds_overlap))
    iou = np.zeros(n_frame)
    for i in range(n_frame):
        iou[i] = overlap_ratio(np.array(gt_bb[i]), np.array(result_bb[i]))
    for i in range(len(thresholds_overlap)):
        success[i] = sum(iou > thresholds_overlap[i]) / n_frame
    return success

def display_tracking(idx, image, result_bb, gt=None):

    image = np.array(image)
    cv2.putText(image, 'Frame %d' % idx, (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255),2)
    cv2.rectangle(image, tuple(result_bb[0:2].astype(np.int)), tuple((result_bb[2:4]+result_bb[:2]).astype(np.int)), (0, 255, 0), 2)
    if gt is not None:
        cv2.rectangle(image, tuple(gt[0:2].astype(np.int)), tuple((gt[2:4]+gt[:2]).astype(np.int)), (0,255,0), 2)

    cv2.imshow('Tracking',image)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        exit()

def get_axis_aligned_bbox(region):
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    x = cx - w/2
    y = cy - h/2
    return x, y, w, h

class InterResultShow:

    def __init__(self, image, pred_box, frame_idx, roi_patch, orig_response):
        gs = gridspec.GridSpec(5, 4+config.n_online_updates)
        self.fig = plt.figure()
        mng = plt.get_current_fig_manager()
        #mng.window.showMaximized()
        # mng.window.state('zoomed')
        axes1 = self.fig.add_subplot(gs[0:3, 0:4])
        self.img_show = plt.imshow(image)
        self.rect_show = plt.Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3], linewidth=3,
                                  edgecolor="red", zorder=1, fill=False)
        axes1.add_patch(self.rect_show)
        self.tx_show = plt.text(5, 40, 'Frame: ' + str(frame_idx), color='yellow', fontsize=20)
        plt.axis('off')
        plt.title('Tracking Result')

        self.fig.add_subplot(gs[3:5, 0:2])
        roi_patch_unpre = un_preprocess_image(roi_patch)
        self.roi_patch_show = plt.imshow(roi_patch_unpre)
        plt.axis('off')
        plt.title('ROI patch')

        self.fig.add_subplot(gs[3:5, 2:4])
        self.response_show = plt.imshow(orig_response)
        plt.axis('off')
        plt.title('Original response')

        self.samples_show = []
        self.text_list = []
        for i in range(config.n_online_updates):
            for j in range(config.n_update_batch):
                self.fig.add_subplot(gs[j, 4+i])
                self.samples_show.append(plt.imshow(np.zeros(roi_patch_unpre.shape, dtype=np.uint8)))
                # if config.reweight:
                #     text_show = plt.text(5, 50, '%0.4f' % 0, color='yellow', fontsize=10)
                #     self.text_list.append(text_show)
                plt.axis('off')
        plt.pause(1)
        self.stop = False
        self.pause = False

    def on_key_event(self, event):
        key = event.key
        if key in 'q':
            plt.close('all')
            self.stop = True
        elif key in 'p':
            self.pause = not self.pause
        while self.pause:
            plt.pause(1)

    def display_inter_result(self, frame_idx, image, roi_patch, orig_response):

        if image.ndim != 3:
            image = np.expand_dims(image, 2)
            image = np.repeat(image,3,2)

        self.img_show.set_data(image)
        self.frame_idx = frame_idx
        self.tx_show.set_text('Frame: ' + str(frame_idx))
        self.roi_patch_show.set_data(roi_patch)
        self.response_show.set_data(orig_response)

    def display_bbox(self, pred_box, seq_name=None, is_save=False):

        self.rect_show.set_xy(pred_box[0:2])
        self.rect_show.set_width(pred_box[2])
        self.rect_show.set_height(pred_box[3])

        plt.pause(0.01)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_event)

        if is_save:
            save_dir = os.path.join('./tracking/inter_results',seq_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join(save_dir, '%04d.png' % self.frame_idx))

    def display_examples(self, patches):

        count = 0
        for i, patch in enumerate(patches):
            self.samples_show[count].set_data(patch)
            count += 1