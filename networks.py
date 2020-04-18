# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import os
import config
import scipy.io
import numpy as np
from utils import generate_delta_bboxes

class FeatureExtractor(nn.Module):
    def __init__(self, model_dir=None):
        super(FeatureExtractor, self).__init__()
        self.feat_extractor = VGG16()
        if model_dir is not None:
            self.load_mat_model(os.path.join(model_dir, 'vgg-16.mat'))

    def forward(self, x):
        x = self.feat_extractor(x)
        return x

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.feat_extractor.load_state_dict(state_dict, strict=False)

    def load_mat_model(self, matfile):
        from_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]
        to_indices = [0, 2, 6, 8, 12, 14, 16, 18, 20, 22]
        mat = scipy.io.loadmat(matfile)
        mat_features = list(mat['layers'])[0]
        # copy conv weights
        for i in range(len(from_indices)):
            from_i = from_indices[i]
            to_i = to_indices[i]
            weight, bias = mat_features[from_i]['weights'].item()[0]
            self.feat_extractor.features[to_i].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.feat_extractor.features[to_i].bias.data = torch.from_numpy(bias[:, 0])


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('relu1_2', nn.ReLU()),
            ('pad1_2', nn.ZeroPad2d((0, 1, 0, 1))),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('relu2_2', nn.ReLU()),
            ('pad2_2', nn.ZeroPad2d((0, 1, 0, 1))),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

            ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            ('relu3_1', nn.ReLU()),
            ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu3_2', nn.ReLU()),
            ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu3_3', nn.ReLU()),

            ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)),
            ('relu4_1', nn.ReLU()),
            ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            ('relu4_2', nn.ReLU()),
            ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),
            ('relu4_3', nn.ReLU()),
        ]))

    def forward(self, input):
        return self.features(input)


class MetaNet(nn.Module):

    def __init__(self, cf_size, out_channels):
        super(MetaNet, self).__init__()
        fw = cf_size[0]
        fh = cf_size[1]
        self.conv = nn.Conv2d(config.feat_channels, config.cf_channels, kernel_size=1, stride=1, padding=0)
        self.cf = nn.Conv2d(config.cf_channels, out_channels, kernel_size=(fh, fw), stride=1, padding=(fh // 2, fw // 2))

    def get_flat_params(self):
        params = []

        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for submodule in module:
                    params.append(submodule._parameters['weight'].view(-1))
                    params.append(submodule._parameters['bias'].view(-1))
            else:
                params.append(module._parameters['weight'].view(-1))
                params.append(module._parameters['bias'].view(-1))

        return torch.cat(params)


class MetaLRNet(MetaNet):

    def __init__(self, cf_size, out_channels):

        super(MetaLRNet, self).__init__(cf_size, out_channels)
        self.weight_init()

    def weight_init(self):
        self.conv.bias.data.fill_(config.meta_lr_init)
        self.cf.bias.data.fill_(config.meta_lr_init)
        self.conv.weight.data.fill_(config.meta_lr_init)
        self.cf.weight.data.fill_(config.meta_lr_init)


class MetaCFNet(MetaNet):

    def __init__(self, cf_size, out_channels):

        super(MetaCFNet, self).__init__(cf_size, out_channels)
        self.weight_init(cf_size[1]//2, cf_size[0]//2)

    def weight_init(self, rh, rw):
        self.conv.bias.data.zero_()
        self.cf.bias.data.zero_()
        self.conv.weight.data = (self.conv.weight.data.normal_() / np.sqrt(self.conv.in_channels)) / 1e5
        self.cf.weight.data = (self.cf.weight.data.normal_() / np.sqrt(self.cf.in_channels * rh * rw)) / 1e5


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, pred_bbox, gt_bbox, anchors):
        delta_bbox, mask = generate_delta_bboxes(anchors, gt_bbox)
        pos_pred_bbox = torch.masked_select(pred_bbox.view(-1, 4), mask)
        pos_target_bbox = torch.masked_select(delta_bbox, mask)
        reg_loss = F.smooth_l1_loss(pos_pred_bbox, pos_target_bbox)
        return reg_loss


class L2normLoss(nn.Module):
    def __init__(self):
        super(L2normLoss, self).__init__()

    def forward(self, pred_map, label_map):
        abs_diff = (pred_map - label_map).abs()
        mask = (abs_diff >= 0.1).float()
        n_effect = mask.sum() + 1 # 1 for divide-by-zero error
        loss = label_map.exp() * mask * (pred_map - label_map).pow(2)
        avg_loss = loss.sum()/n_effect
        return avg_loss