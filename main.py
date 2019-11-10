import argparse
import os
import random
import shutil
import time
import warnings
import logging
import copy

import numpy as np
import pandas as pd

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
from src.slice_nets_2d import get_dense2d_unet_v1, get_dense2d_unet_v1_quantized
from src.lung_seg_v1 import get_customized_dataloader, MemoryDataset_v1
from src.loss import weighted_bce_dice_loss_2d
from src.lr_scheduler import CosineWithRestarts
from metrics import binary_dice
from utils.meters import AverageMeter, MaxMeter, ProgressMeter, CumMeter


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
parser.add_argument('--quantize', dest='quantize', action='store_true',
                    help='using quantization-aware traing')

log_path = 'logs/lung_segment.log'
logger = configure(log_path, logging.INFO)
best_dice = 0

@log_arguments(logger)
def update_args(args):
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
        # main_worker(args.gpu, ngpus_per_node, args)
        main_worker(args)

def main_worker(args):
    global best_dice
    model = get_dense2d_unet_v1()
    model = get_dense2d_unet_v1_quantized()

    # model = resume(model, args)

    if args.quantize:
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
   
    cudnn.benchmark = True

    train_dataloader = get_customized_dataloader(split='trn',
    # train_dataloader = MemoryDataset_v1(
                                    data_dir='/home/xyh/data/PreData/LUNA16/LUNA16_Original_Lung/lung_val.npy',
                                    mask_dir='/home/xyh/data/PreData/LUNA16/LUNA16_Original_Lung/mask_val.npy',
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
    )
    valid_dataloader = get_customized_dataloader(split='val',
    # valid_dataloader = MemoryDataset_v1(
                                    data_dir='/home/xyh/data/PreData/LUNA16/LUNA16_Original_Lung/lung_val.npy',
                                    mask_dir='/home/xyh/data/PreData/LUNA16/LUNA16_Original_Lung/mask_val.npy',
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
    )


    criterion = weighted_bce_dice_loss_2d()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    scheduler = CosineWithRestarts(optimizer,
                     last_epoch=-1,
                     T_max=10,
                     T_mult=1,
                     eta_min=0.000001,
                     decay=0.1,
                     start_decay_cycle=2)

    model = model.cuda()

    if args.evaluate:
        validate(valid_dataloader, model, criterion, 0, args)
        return

    for epoch in range(args.epochs):
        train(train_dataloader, model, criterion, optimizer, epoch, args)

        if epoch > 4 and args.quantize:
            # Freeze quantizer parameters
            model.apply(torch.quantization.disable_observer)
        if epoch > 2 and args.quantize:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        if args.quantize:
            qat_model = torch.quantization.convert(model.eval(), inplace=False)
            dice = validate(valid_dataloader, qat_model, criterion, epoch, args)
        else:       
            dice = validate(valid_dataloader, model, criterion, epoch, args)

        scheduler.step()

        is_best = dice > best_dice
        best_dice = max(dice, best_dice)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            # if hasattr(model, 'module'):
                # model_ = copy.deepcopy(model.module)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_metric': best_dice,
                'optimizer' : optimizer.state_dict(),
            }, 
            is_best,
            filename='checkpoints/checkpoint_{epoch}.pth.tar'.format(epoch=epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_metric': best_dice,
                'optimizer' : optimizer.state_dict(),
            }, 
            is_best,
            filename='checkpoints/qat_checkpoint_{epoch}.pth.tar'.format(epoch=epoch))

def resume(model, args):
    global best_dice
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint)    
            # args.start_epoch = checkpoint['epoch']
            # best_dice = checkpoint['best_acc1']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            # model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    return model

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = CumMeter('Time(s)', ':.2f')
    data_time = CumMeter('Data(s)', ':.2f')
    losses = AverageMeter('bce', ':.3e')
    dices = AverageMeter('dice', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dices],
        prefix="Epoch: [{}][trn]".format(epoch),
        )

    model.train()
    # train_loader.shuffle()
    end = time.time()
    for batch, (x, y1, y2) in enumerate(train_loader): 
    # BUG: RuntimeError: cuDNN error: CUDNN_STATUS_BAD_PARAM           
    # for batch in range(0, len(train_loader)//args.batch_size):   
    #     x, y1, y2 = train_loader.get_batch(range(batch * args.batch_size, batch * (args.batch_size+1)))         

        data_time.update(time.time() - end)

        x, y1, y2 = x.cuda(), y1.cuda(), y2.cuda()
        out1, out2 = model(x)
        loss1 = criterion(out1, y1)
        loss2 = criterion(out2, y2)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out_pred = (out1>out2).float()*out1 + (out2>out1).float()*out2
        out_gt = y1 + y2  
        dice = binary_dice(out_pred, out_gt)

        losses.update(loss.item(), 2)
        dices.update(dice.item(), 2)
        batch_time.update(time.time() - end)
        end = time.time()

        del loss1, loss2, loss
        del out1, out2, out_gt, out_pred
        if batch % args.print_freq == 0:
            progress.display(batch, logging.DEBUG)

    progress.display(batch, logging.INFO, reduce=True)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = CumMeter('Time(s)', ':.2f')
    data_time = CumMeter('Data(s)', ':.2f')
    losses = AverageMeter('bce', ':.3e')
    dices = AverageMeter('dice', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, dices],
        prefix="Epoch: [{}][val]".format(epoch),
        )

    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch, (x, y1, y2) in enumerate(val_loader):
        # for batch in range(0, len(val_loader)//args.batch_size):   
        #     x, y1, y2 = val_loader.get_batch(range(batch * args.batch_size, batch * (args.batch_size+1)))  

            data_time.update(time.time() - end)

            x, y1, y2 = x.cuda(), y1.cuda(), y2.cuda()
            out1, out2 = model(x)
            loss1 = criterion(out1, y1)
            loss2 = criterion(out2, y2)
            loss = loss1 + loss2

            out_pred = (out1>out2).float()*out1 + (out2>out1).float()*out2
            out_gt = y1 + y2  
            dice = binary_dice(out_pred, out_gt)

            losses.update(loss.item(), 2)
            dices.update(dice.item(), 2)
            batch_time.update(time.time() - end)
            end = time.time()

            del loss1, loss2, loss
            del out1, out2, out_gt, out_pred
            if batch % args.print_freq == 0:
                progress.display(batch, logging.DEBUG)

        progress.display(batch, logging.INFO, reduce=True)

    return dice.cpu().data

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')

if __name__ == '__main__':
    import pdb
    main()
    # pdb.set_trace()
