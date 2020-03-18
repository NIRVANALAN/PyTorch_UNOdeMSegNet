import string
import random
from segmentation_pytorch.models import create_model
from segmentation_pytorch.utils.metrics import MIoUMetric, MPAMetric
from segmentation_pytorch.utils.losses import PixelCELoss
from data import build_val_loader, build_train_loader, build_inference_loader
from easydict import EasyDict
import argparse
from collections import OrderedDict
import yaml
import copy
import time
import numpy as np
import torch.nn as nn
from warmup_scheduler import GradualWarmupScheduler
import segmentation_pytorch as smp
import torch
import os
import sys
import warnings
# from torchvision.models.segmentation import deeplabv3_resnet50

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
# import segmentation_models_pytorch as smp


def random_string(stringLength=4):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def main(args):
    num_classes = args.model.get('num_classes', 8)
    model = create_model(args)
    num_res = args.model.get('num_res', None)
    if num_res:
        assert num_res >= 2
    print(f'multi_stage_UNET: {num_res}')

    # load class weight
    class_weight = None
    if hasattr(args.loss, 'class_weight'):
        if isinstance(args.loss.class_weight, list):
            class_weight = np.array(args.loss.class_weight)
            print(
                f'Loading class weights from static class list: {class_weight}')
        elif isinstance(args.loss.class_weight, str):
            try:
                class_weight = np.load(args.loss.class_weight)
                class_weight = np.divide(
                    1.0, class_weight, where=class_weight != 0)
                class_weight /= np.sum(class_weight)
                print(
                    f'Loading class weights from file {args.loss.class_weight}')
            except OSError as e:
                print(f'Error cannot open class weight file, {e}, exiting')
                exit(-1)

    criterion = PixelCELoss(
        num_classes=num_classes,
        weight=class_weight, multi_stage=num_res)
    metrics = [  # TODO, check IOU calculation, should ignore some classes?
        MIoUMetric(num_classes=num_classes, ignore_index=None),
        MPAMetric(num_classes=num_classes,
                  ignore_index=None)  # ignore background
    ]

    torch.cuda.set_device(args.gpu)
    device = 'cuda'
    lr = args.train.lr
    # optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=args.train.get('weight_decay',
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=args.train.get('weight_decay',
                                                                                               0.0005),)
    # dataset
    args.pixel_ce = True  # TODO add DICE LOSS
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

    max_miou = 0.
    num_epochs = args.get('epochs', 200)
    # load checkpoint
    pretrained_path = args.model.get('pretrained', None)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
        print('Path to the checkpoint of pretrained:', pretrained_path)
        if args.model.get('load_optim', True):
            path_to_optimizer = pretrained_path.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            optimizer.load_state_dict(torch.load(path_to_optimizer))
        else:
            warnings.warn('ignoring optimizer, set "load_optim: True" to load')
    # load finish
    print(f'training cfg: {args.train}')
    if args.train.get('warm_up', False):
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_epochs)
        scheduler = GradualWarmupScheduler(optimizer, total_epoch=args.train.get('warm_up', 10),
                                           multiplier=args.train.get(
                                               'warm_up_multiplier', 8),
                                           after_scheduler=scheduler_cosine)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=args.train.lr_gamma,
                                                         milestones=args.train.lr_iters)
        # from torch.optim.lr_scheduler import StepLR
        # scheduler = StepLR(optimizer, step_size=40,
        #                    gamma=0.1, last_epoch=num_epochs)
    save_model_iter = args.train.save_iter

    test_slides = args.get('test_slides', ['S1_Helios_1of3_v1270.tiff', 'NA_T4_122117_01.tif',
                                           'NA_T4R_122117_19.tif', ])
    test_loader = {}
    model_log = {}
    save_prefix = random_string(4)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    print(f'save prefix: {save_prefix}')
    for test_slide in test_slides:
        # print(f'inference {test_slide}')
        args.data.slide_name = test_slide
        args.eval = True
        tiff_loader, tiff_dataset, shape = build_inference_loader(args)
        test_loader[test_slide] = tiff_loader
    for i in range(1, num_epochs):
        print('\nEpoch: {}  lr: {}'.format(i, optimizer.param_groups[0]['lr']))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # ================= save model and delete old models ===================== #
        model_log[i] = {'miou': f'{valid_logs["miou"].mean():.4}',
                        'mpa': f'{valid_logs["mpa"].mean(): .4}'}
        if max_miou < float(model_log[i]['miou']):
            max_miou = float(model_log[i]['miou'])  # mean
        torch.save(model.state_dict(), os.path.join(args.save_path, f'{save_prefix}_model_{i}'
                                                                    f'_{valid_logs["miou"]:.4}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'{save_prefix}_optimizer_{i}'
                                                                        f'_{valid_logs["miou"]:.4}.pth'))
        print(
            f'Model saved: MIOU:{valid_logs["miou"]}, MPA:{valid_logs["mpa"]}, best_miou: {max_miou}')
        if i > save_model_iter and float(model_log[i - save_model_iter]['miou']) < max_miou:
            os.remove(
                os.path.join(args.save_path, f'{save_prefix}_model_{i - save_model_iter}'
                             f'_{model_log[i - save_model_iter]["miou"]}.pth'))
            os.remove(
                os.path.join(args.save_path, f'{save_prefix}_optimizer_{i - save_model_iter}'
                             f'_{model_log[i - save_model_iter]["miou"]}.pth'))
            print(
                f'delete model: {i - save_model_iter}_{model_log[i - save_model_iter]["miou"]}.pth')

        # ================= update lr ===================== #
        scheduler.step()
        # ================= test on testset ===================== #
        """
		if i % 10 == 9 or i == num_epochs - 1:
			print(f'epoch :{num_epochs}, testing on 3 slides')
			test_log = {'miou': np.array([]), 'mpa': np.array([])}
			for test_slide in test_slides:
				test_logs = valid_epoch.run(test_loader[test_slide])
				print(f'{test_slide}, miou:{test_logs["miou"]}, mpa:{test_logs["mpa"]}')
				test_log['miou'] = np.append(test_log['miou'], test_logs['miou'])
				test_log['mpa'] = np.append(test_log['mpa'], test_logs['mpa'])
		"""
    print(f'save prefix: {save_prefix}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax classification loss")

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--config', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    params = EasyDict(config)
    params.seed = args.seed
    params.local_rank = args.local_rank
    params.gpu = args.gpu
    main(params)
