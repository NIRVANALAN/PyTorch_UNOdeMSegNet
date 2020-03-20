from model import WaveletModel
from segmentation_pytorch.utils.losses import WNetLoss
from segmentation_pytorch.utils.metrics import FWAVACCMetric, MPAMetric
from data import build_val_loader, build_train_loader
from easydict import EasyDict
import argparse
from collections import OrderedDict
import yaml
import copy
import time
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import segmentation_pytorch as smp
import torch
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
# import segmentation_models_pytorch as smp


def main(args):
    num_classes = args.model.get('num_classes', 8)
    if args.data.get('wavelet', False):
        model = WaveletModel(
            encoder_name=args.model.arch,
            # encoder_weights='imagenet',
            encoder_weights=None,
            activation='softmax',  # whatever, will do .max during metrics calculation
            classes=num_classes)
    else:
        model = smp.Wnet(
            encoder_name_1=args.model.arch,
            # encoder_weights='imagenet',
            activation='softmax',  # whatever, will do .max during metrics calculation
            classes=num_classes)
        pass

    metrics = [
        # MIoUMetric(num_classes=num_classes, ignore_index=0),
        FWAVACCMetric(num_classes=num_classes, ignore_index=None),
        MPAMetric(num_classes=num_classes,
                  ignore_index=None)  # ignore background
    ]

    device = 'cuda'
    lr = args.train.lr
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': lr},

        # decrease lr for encoder in order not to permute
        # pre-trained weights with large gradients on training start
        {'params': model.encoder.parameters(), 'lr': lr},
    ])
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.train.lr, momentum=args.train.momentum)
    # dataset
    criterion = WNetLoss
    train_loader, _ = build_train_loader(args)
    valid_loader, _ = build_val_loader(args)

    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0

    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.train.lr_iters, gamma=0.1)

    num_epochs = args.get('epochs', 100)

    for i in range(0, num_epochs):

        print('\nEpoch: {}  lr: {}'.format(i, optimizer.param_groups[0]['lr']))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        exp_lr_scheduler.step()

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['miou']:
            max_score = valid_logs['miou']  # mean
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            torch.save(
                model,
                os.path.join(
                    args.save_path,
                    f'./{max_score}.pth'))
            print(
                f'Model saved: MIOU:{valid_logs["miou"]}, MPA:{valid_logs["mpa"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.local_rank = args.local_rank
    main(params)
