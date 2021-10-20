# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Additionally modified by Hwanjun Song for MEDUSA

import os
import sys
import datetime
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
import util.misc as utils
from engine import evaluate, train_one_epoch
from models import build_model

import argparse
from arguments import get_args_parser


def main(args):

    # For Data Loader
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # gradient accumulation setup
    if args.n_iter_to_acc > 1:

        if args.batch_size % args.n_iter_to_acc != 0:
            print("Not supported divisor for acc grade.")
            sys.exit(1)

        print("Gradient Accumulation is applied.")
        print("The batch: ", args.batch_size, "->", int(args.batch_size / args.n_iter_to_acc),
              'but updated every ', args.n_iter_to_acc, 'steps.')
        args.batch_size = args.batch_size / args.n_iter_to_acc
    ###

    # distributed data parallel setup
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors, param_dicts = build_model(args)
    model.to(device)

    # parallel model setup
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # print parameter info.
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_scratch_parameters = sum([p.numel() for p in param_dicts[0]['params']])
    n_finetune_parameters = sum([p.numel() for p in param_dicts[1]['params']])
    print('num of total trainable prams:' + str(n_parameters))
    print('num of trainable prams from scratch:' + str(n_scratch_parameters))
    print('num of trainable prams from pretrained model:' + str(n_finetune_parameters))

    # optimizer setup
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # build data loader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    print("Train Data:", len(dataset_train))
    print("Test Data:", len(dataset_val))

    # data samplers
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco_reader.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # check resume
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    output_dir = Path(args.output_dir)

    # only evaluation purpose
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device)

        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # specify the current epoch number for samplers
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # training one epoch with default setting
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, print_freq=args.print_freq, n_iter_to_acc=args.n_iter_to_acc)
        lr_scheduler.step()

        # model save
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # eval
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, print_freq=args.print_freq,
        )

        # logs
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    # Load default parameters for MEDUSA
    print(torch.__version__)
    parser = argparse.ArgumentParser('MEDUSA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # log file name
    args.output_dir = ''
    args.output_dir += args.backbone + '-'
    args.output_dir += str(args.epochs) + '-'
    args.output_dir += str(args.batch_size)

    args.output_dir = 'MEDUSA-' + args.backbone + '-batch-' + \
                      str(args.batch_size) + '-epoch-' + str(args.epochs)

    # make log directories
    if args.output_dir:
        log_main = 'logs'
        args.output_dir = os.path.join(log_main, args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print('log', args.output_dir)

    main(args)
