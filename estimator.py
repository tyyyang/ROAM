import config
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from networks import FeatureExtractor
from roam import ROAM

class Estimator():

    def __init__(self, train_data, eval_data):

        self.train_data = train_data
        self.eval_data = eval_data

        feat_extractor = FeatureExtractor(config.feat_dir)
        if torch.cuda.is_available():
            feat_extractor.cuda()

        # clear logs before training
        for root, dirs, files in os.walk(config.log_dir + '/roam', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        self.writer = SummaryWriter(config.log_dir + '/roam')
        self.roam = ROAM(feat_extractor, self.writer)
        if torch.cuda.is_available():
            self.roam.cuda()

    def train_roam(self, args):

        self.roam.train()
        mi_optimizer = torch.optim.Adam(self.roam.meta_init.parameters(), lr=args.lr_meta_init)
        mo_optimizer = torch.optim.Adam(self.roam.meta_opti.parameters(), lr=args.lr_meta_opti)
        mi_scheduler = torch.optim.lr_scheduler.StepLR(mi_optimizer, config.n_decay, gamma=config.lr_decay)
        mo_scheduler = torch.optim.lr_scheduler.StepLR(mo_optimizer, config.n_decay, gamma=config.lr_decay)

        train_data_loader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        train_size = len(self.train_data)
        n_iter_per_epoch = train_size//args.batch_size

        if args.mGPUs:
            self.roam = nn.DataParallel(self.roam)

        n_total_iter = 0
        for epoch in range(args.max_epochs):
            train_data_iter = iter(train_data_loader)
            for i in range(n_iter_per_epoch):
                n_total_iter = epoch * n_iter_per_epoch + i + 1
                tic = time.time()
                patches, label_maps, gt_bboxes = next(train_data_iter)
                if torch.cuda.is_available():
                    patches = patches.cuda()
                    label_maps = label_maps.cuda()
                    gt_bboxes = gt_bboxes.cuda()
                map_loss, bbox_loss, meta_loss = self.roam(patches, label_maps, gt_bboxes, n_total_iter)
                map_loss = map_loss.mean()
                bbox_loss = bbox_loss.mean()
                meta_loss = meta_loss.mean()
                mi_optimizer.zero_grad()
                mo_optimizer.zero_grad()
                meta_loss.backward()
                meta_init_params = self.roam.module.meta_init.parameters() if args.mGPUs else self.roam.meta_init.parameters()
                for param in meta_init_params:
                    param.grad.data.clamp_(-10, 10)
                meta_opti_params = self.roam.module.meta_opti.parameters() if args.mGPUs else self.roam.meta_opti.parameters()
                for param in meta_opti_params:
                    param.grad.data.clamp_(-1, 1)
                mi_optimizer.step()
                mo_optimizer.step()
                print('Iter:%d, map_loss:%.4f, bbox_loss:%.4f, meta_loss:%.4f'%
                      (n_total_iter, map_loss.data.item(), bbox_loss.data.item(), meta_loss.data.item()), end='')
                if (n_total_iter) % config.disp_inter == 0:
                    self.writer.add_scalar('roam_training/map_loss', map_loss.data.item(), n_total_iter)
                    self.writer.add_scalar('roam_training/bbox_loss', bbox_loss.data.item(), n_total_iter)
                    self.writer.add_scalar('roam_training/meta_loss', meta_loss.data.item(), n_total_iter)
                    self.writer.add_scalar('roam_training/mi_lr', mi_scheduler.get_lr()[0], n_total_iter)
                    self.writer.add_scalar('roam_training/mo_lr', mo_scheduler.get_lr()[0], n_total_iter)

                toc = time.time() - tic
                print(' Time %.3f' % toc)

            mi_scheduler.step()
            mo_scheduler.step()

            if (epoch+1) % args.evaluate_epochs == 0:
                if not os.path.isdir(config.model_dir):
                    os.makedirs(config.model_dir)

                file_name = 'roam_epoch{:02d}.torch'.format(epoch+1)
                save_state = self.roam.module.state_dict() if args.mGPUs else self.roam.state_dict()
                torch.save(save_state, os.path.join(config.model_dir, file_name))
                print("Save model to %s" % config.model_dir)

                print('Evaluate for epoch {}'.format(epoch+1))
                self.evaluate_roam(n_total_iter, args)

    def evaluate_roam(self, n_total_iter, args):

        self.roam.eval()
        map_loss_total, bbox_loss_total, meta_loss_total = 0, 0, 0
        eval_batch_size = args.batch_size
        eval_data_loader = DataLoader(self.eval_data, batch_size=eval_batch_size, shuffle=False, num_workers=8)
        eval_data_iter = iter(eval_data_loader)
        eval_size = len(self.eval_data)
        n_iter_eval = eval_size // eval_batch_size

        for i in range(n_iter_eval):
            patches, label_maps, gt_bboxes = next(eval_data_iter)
            if torch.cuda.is_available():
                patches = patches.cuda()
                label_maps = label_maps.cuda()
                gt_bboxes = gt_bboxes.cuda()
            map_loss, bbox_loss, meta_loss = self.roam(patches, label_maps, gt_bboxes, n_total_iter)
            map_loss_total += map_loss.mean().data.item()
            bbox_loss_total += bbox_loss.mean().data.item()
            meta_loss_total += meta_loss.mean().data.item()
            print('Evaluation---Iteration %d: label_loss: %.4f, bbox_loss: %.4f,  meta_loss: %.4f' %
                  (i + 1, map_loss_total/(i+1), bbox_loss_total/(i+1), meta_loss_total / (i + 1)), end='')
            print('')

        map_loss_avg = map_loss_total/n_iter_eval
        bbox_loss_avg = bbox_loss_total/n_iter_eval
        meta_loss_avg = meta_loss_total/n_iter_eval
        self.writer.add_scalar('roam_evaluation/map_loss', map_loss_avg, n_total_iter)
        self.writer.add_scalar('roam_evaluation/bbox_loss', bbox_loss_avg, n_total_iter)
        self.writer.add_scalar('roam_evaluation/meta_loss', meta_loss_avg, n_total_iter)
        print('Evaluation average meta_loss: %.4f' % meta_loss_avg, end='')
        print('\n')
        self.roam.train()
