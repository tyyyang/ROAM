# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from utils import preprocess_gradients
from networks import MetaLRNet, MetaCFNet
import config

class MetaInitializer(nn.Module):

    def __init__(self):
        super(MetaInitializer, self).__init__()
        self.meta_cfnet_lr = MetaLRNet(config.base_filter_size, 1)
        self.meta_cfnet_param = MetaCFNet(config.base_filter_size, 1)
        self.meta_regnet_lr = MetaLRNet(config.base_filter_size, 4)
        self.meta_regnet_param = MetaCFNet(config.base_filter_size, 4)

    def initialize(self):
        cf_lr = self.meta_cfnet_lr.get_flat_params()
        cf_param = self.meta_cfnet_param.get_flat_params()
        reg_lr = self.meta_regnet_lr.get_flat_params()
        reg_param = self.meta_regnet_param.get_flat_params()
        return torch.cat([cf_param, reg_param], 0), torch.cat([cf_lr, reg_lr], 0)

class MetaOptimizer(nn.Module):
    """ Baseline: recursively update lrs using lstm
    lr_t = lstm(lr_t-1)
    theta_t = theta_t-1 - lr_t * grad_t
    """

    def __init__(self):
        super(MetaOptimizer, self).__init__()

        self.cw_lstm = nn.LSTM(input_size=5, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_layer_num)
        self.linear = nn.Linear(config.lstm_hidden_size, 1)
        self.linear.bias.data[0] = -10

    def forward(self, x):
        self.cw_lstm.flatten_parameters()
        output, (self.h0, self.c0) = self.cw_lstm(x.unsqueeze(0), (self.h0, self.c0))
        x = torch.sigmoid(self.linear(output.squeeze(0)))
        return x

    def initialize(self, init_lrs):
        self.lrs = init_lrs[:, None]
        batch_size = init_lrs.shape[0]
        self.h0 = torch.zeros(config.lstm_layer_num, batch_size, config.lstm_hidden_size)
        self.c0 = torch.zeros(config.lstm_layer_num, batch_size, config.lstm_hidden_size)
        if torch.cuda.is_available():
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()

    def meta_update(self, flat_params, loss, flat_grads, writer=None, iter_step=0):

        loss = loss.expand_as(flat_grads)
        if not self.training:
            self.h0 = self.h0.detach()
            self.c0 = self.c0.detach()
            self.lrs = self.lrs.detach()
        flat_params = flat_params.detach()
        inputs = torch.cat((self.lrs.detach(), preprocess_gradients(flat_grads),
                            flat_params[:, None].detach(), loss[:, None].detach()), 1)
        self.lrs = self(inputs)
        # Meta update itself
        new_params = flat_params - self.lrs.view(-1) * flat_grads

        if self.training and iter_step % config.disp_inter == 0 and torch.cuda.current_device() == 0 and writer is not None:
            offset = (config.feat_channels + 1) * config.cf_channels + \
                     config.base_filter_size[0] * config.base_filter_size[1] * config.cf_channels + 1
            writer.add_histogram('roam_training/update_cf_params', new_params[:offset], iter_step)
            writer.add_histogram('roam_training/update_cf_lrs', self.lrs[:offset], iter_step)
            writer.add_histogram('roam_training/update_reg_params', new_params[offset:], iter_step)
            writer.add_histogram('roam_training/update_reg_lrs', self.lrs[offset:], iter_step)

        return new_params