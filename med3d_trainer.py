import argparse
import os
import sys
import random
import shutil
import time
import warnings
import logging
import copy
import pdb

warnings.filterwarnings("ignore")
                        
import numpy as np
import pandas as pd
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils.log import configure, log_arguments
from lib.models import resnet as models
from lib.datasets.med3d import MED3D_dataset
from lib.metrics import binary_dice
from utils.meters import AverageMeter, MaxMeter, ProgressMeter, CumMeter

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

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
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--quantize', dest='quantize', action='store_true',
#                     help='using quantization-aware traing')

parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')




parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    dest='multiprocessing_distributed',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# logging config
parser.add_argument('--log-level', default=20, type=int,
                    help='logging level')

trainer_name = sys.argv[0].replace('.py','')
time_string = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
log_path = 'logs/'+trainer_name+'_{}.log'.format(time_string)
writer_path = 'runs/'+trainer_name+'/{}'.format(time_string)


best_metric = 0
metric_type = 'cross_entropy'

logger = configure(log_path, logging.INFO)

@log_arguments(logger)
def update_args(args):
    if not torch.cuda.is_available() or args.gpu==-1:
        args.device = torch.device('cpu')
    elif args.gpu is None:
        args.device = torch.device('cuda')
    else:
        loc = 'cuda:{}'.format(args.gpu)
        args.device = torch.device(loc)
    return args

def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8022'

    args = parser.parse_args()
    args = update_args(args)
    logging.debug('running on device: {}.'.format(args.device))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        logging.warning('You have chosen a specific GPU. This will completely '
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
        # main_worker(args)

def main_worker(gpu, ngpus_per_node, args):
    global best_metric

    writer = SummaryWriter(writer_path)
    model = models.__dict__[args.arch](sample_input_D=256,
                            sample_input_H=256,
                            sample_input_W=256,
                           num_seg_classes=3)
#     pdb.set_trace()
    if args.resume:
        model, parameters = resume(model, args)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
            
    model.to(args.device)
    logging.info('load model.')

    if args.gpu == -1:
        model = model
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).to(args.device)

    if args.gpu != -1:
        cudnn.benchmark = True

#     writer.add_graph(model, torch.randn(1,1,256,512,512).to(args.device))
#     writer.flush()
                     
    train_dataset = MED3D_dataset('trn', args.data_dir)
    val_dataset = MED3D_dataset('val', args.data_dir)
    logging.info('load dataset.')
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device)
    params = [
            { 'params': parameters['base_parameters'], 'lr': args.lr }, 
            { 'params': parameters['new_parameters'], 'lr': args.lr*100 }
            ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)   
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    logging.info('config criterion, optimizer, scheduler.')
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return
    
    
    for epoch in range(args.epochs):
        logging.info('lr = {}'.format(scheduler.get_lr()))
        
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        loss, dice = validate(val_loader, model, criterion, epoch, args, writer)

        is_best = dice < best_metric
        best_metric = min(dice, best_metric)

#         writer.add_scalar('val_dice', dice)
#         writer.flush()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if hasattr(model, 'module'):
                model_ = copy.deepcopy(model.module)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_metric': best_metric,
                'metric_type': metric_type,
                'optimizer' : optimizer.state_dict(),
            }, 
            is_best,
            filename='{path}/checkpoint_{epoch}.pth.tar'.format(path=writer_path, epoch=epoch))
        scheduler.step()

def resume(model, args):
    global best_metric

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=args.device)
#             new_state_dict = OrderedDict()
#             for k,v in ckpt['state_dict'].items():
#                 _k = k.replace('module.','')
#                 new_state_dict[_k] = v

            # partially load state dict
            model_dict = model.state_dict()
            pretrained_state = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_state)
            model.load_state_dict(model_dict)    
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
            logging.warning("=> no checkpoint found at '{}'".format(args.resume))

        new_parameters = [] 
        for pname, p in model.named_parameters():
            if pname not in pretrained_state.keys():
                new_parameters.append(p)

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                    'new_parameters': new_parameters}

        return model, parameters
    return model, model.parameters()

def train(train_loader, model, criterion, optimizer, epoch, args, writer=None):
    # TODO: get writer from locals()
    # writer = locals()['writer']
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
    end = time.time()
    for batch, (volumes, label_masks) in enumerate(train_loader): 

        data_time.update(time.time() - end)

        volumes, label_masks = volumes.to(args.device), label_masks.to(args.device)
        out_masks = model(volumes)
        
        # resize label
        [n, _, d, h, w] = out_masks.shape
        
        new_label_masks = F.interpolate(label_masks, size=(d,h,w))#.to(torch.int64)
        loss = criterion(out_masks, new_label_masks.squeeze(1).to(torch.int64))

        logging.debug('{}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: add dice
        pred_masks = F.softmax(out_masks, dim=1).argmax(dim=1)
        dice = binary_dice(pred_masks, new_label_masks)

        losses.update(loss.item(), 2)
        dices.update(dice.item(), 2)
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            progress.display(batch, logging.DEBUG)
            writer.add_scalar('Loss/train',
                loss.item(),
                epoch * len(train_loader) + batch)
#             output_grid = select_rand(out_pred[:,1,...].detach(), out_gt[:,1,...].detach())
#             writer.add_image('out_pred', 
#                 output_grid, 
#                 epoch * len(train_loader) + batch,
#                 dataformats='CHW')

    progress.display(batch, logging.INFO, reduce=True)

def validate(train_loader, model, criterion, epoch, args, writer=None):
    batch_time = CumMeter('Time(s)', ':.2f')
    data_time = CumMeter('Data(s)', ':.2f')
    losses = AverageMeter('bce', ':.3e')
    dices = AverageMeter('dice', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, dices],
        prefix="Epoch: [{}][val]".format(epoch),
        )

    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch, (volumes, label_masks) in enumerate(train_loader): 

            data_time.update(time.time() - end)

            volumes, label_masks = volumes.to(args.device), label_masks.to(args.device)
            out_masks = model(volumes)

            new_label_masks = nn.functional.interpolate(label_masks, size=(d,h,w))#.to(torch.int64)
            loss = criterion(out_masks, new_label_masks.squeeze(1).to(torch.int64))

            pred_masks = F.softmax(out_masks, dim=1).argmax(dim=1)
            dice = binary_dice(pred_masks, new_label_masks)

            losses.update(loss.item(), 2)
            dices.update(dice.item(), 2)
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                progress.display(batch, logging.DEBUG)
                writer.add_scalar('Loss/eval',
                    loss.item(),
                    epoch * len(train_loader) + batch)

        progress.display(batch, logging.INFO, reduce=True)

if __name__ == '__main__':
    import pdb
    main()
    # pdb.set_trace()
