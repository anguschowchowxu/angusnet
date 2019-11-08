import argparse
import os
import random
import shutil
import time
import warnings
import logging 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from log import configure, log_arguments
from src.slice_nets_2d import get_dense2d_unet_v1
from src.lung_seg_v1 import get_customized_dataloader
from src.loss import weighted_bce_dice_loss_2d
from src.lr_scheduler import CosineWithRestarts
from metrics import binary_dice
from meters import AverageMeter, MaxMeter, ProgressMeter, CumMeter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--log_level', default=20, type=int,
                    help='logging level')

logger = configure(None, logging.INFO)

@log_arguments(logger)
def update_args(args):
    pass
    return args

def main():
    args = parser.parse_args()
    args = update_args(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker():
    pass

if __name__ == '__main__':

    model = get_dense2d_unet_v1()
    # d = dict(batch_size=1, 
    #             num_workers=1, 
    #             data_dir='/home/xyh/data/PreData/LUNA16/LUNA16_Original_Lung')
    # ds = get_customized_dataloader(split='trn', 
    #                                 ids_list=train_list, 
    #                                 transform=None, 
    #                                 **d)
    opt_loss = weighted_bce_dice_loss_2d()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    scheduler = CosineWithRestarts(optimizer,
                     last_epoch=-1,
                     T_max=10,
                     T_mult=1,
                     eta_min=0.000001,
                     decay=0.1,
                     start_decay_cycle=2)

    model = model.cuda()
    model.train()

    end = time.time()
    for epoch in range(20):

        batch_time = CumMeter('Time(s)', ':6.3f')
        data_time = CumMeter('Data(s)', ':6.3f')
        losses = AverageMeter('bce', ':.4e')
        dices = MaxMeter('dice', ':.4f')
        progress = ProgressMeter(
            40, 5,
            [batch_time, data_time, losses, dices],
            prefix="Epoch: [{}]".format(epoch))

        for batch in range(40):
            x = torch.randn(2,3,512,512).cuda()
            y1 = torch.randint(0,2, (2,3,512,512)).cuda()
            y2 = torch.randint(0,2, (2,3,512,512)).cuda()

            out1, out2 = model(x)
            loss1 = opt_loss(out1, y1)
            loss2 = opt_loss(out2, y2)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_pred = (out1>out2).float()*out1 + (out2>out1).float()*out2
            out_gt = y1 + y2  
            dice = binary_dice(out_pred, out_gt)

            losses.update(loss, 2)
            dices.update(dice, 2)
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % 5 == 0:
                progress.display(batch)