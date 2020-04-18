# ------------------------------------------------------------------
# PyTorch implementation of
#  "ROAM: Recurrently Optimizing Tracking Model", CVPR, 2020
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

from datasets import VIDataset
from estimator import Estimator
import config
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ROAM params')

    # config training
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--eval_epochs', dest='evaluate_epochs',
                        help='number of epochs to evaluate',
                        default=1, type=int)

    # config server running
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=4, type=int)


    # config learning rate
    parser.add_argument('--lr_mi', dest='lr_meta_init',
                        help='learning rate for meta_initializer',
                        default=1e-6, type=float)
    parser.add_argument('--lr_mo', dest='lr_meta_opti',
                        help='learning rate for meta_optimizer',
                        default=1e-3, type=float)

    return parser.parse_args()

def experiment():

    args = parse_args()
    train_data = VIDataset(config.root_dir, is_train=True)
    eval_data = VIDataset(config.root_dir, is_train=False)

    estimator = Estimator(train_data, eval_data)
    estimator.train_roam(args)

if __name__=='__main__':

    experiment()
