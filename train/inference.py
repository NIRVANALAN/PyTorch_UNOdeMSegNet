from tqdm import tqdm
from util.loss import PixelCELoss
from util.metrics import MIoUMetric, MPAMetric
from data import build_inference_loader
from easydict import EasyDict
import argparse
import yaml
import copy
import time
import tifffile as tiff
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import segmentation_models_pytorch as smp
from torchvision import transforms
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())


def evaluated_checkpoint(args):
    valid_loader, test_dataset = build_val_loader(args)
    model = torch.load(args.best_model)
    num_classes = args.model.get('num_classes', 8)
    criterion = PixelCELoss(num_classes=num_classes)
    device = 'cuda'
    metrics = [
        MIoUMetric(num_classes=num_classes, ignore_index=None),
        MPAMetric(num_classes=num_classes,
                  ignore_index=None)  # ignore background
    ]
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True,
    )
    valid_epoch.run(valid_loader)


def main(args):
    # evaluated_checkpoint(args)
    device = 'cuda'
    model = torch.load(args.best_model)
    patch_size = args.data.test_img_size
    tiff_loader, tiff_dataset = build_inference_loader(args)
    pred_mask = np.zeros(shape=(tiff_dataset.get_img_array_shape()))
    for img, (h, w) in tqdm(tiff_loader):
        # pdb.set_trace()
        pr_tile_mask = model.predict(img.to(device))
        pr_tile_mask = pr_tile_mask.argmax(1).cpu().numpy()  # BC*H*W
        for i in range(len(h)):
            x, y = h[i], w[i]
            pred_mask[x * patch_size:(x + 1) * patch_size,
                      y * patch_size:(y + 1) * patch_size] = pr_tile_mask[i]
    slide_name = args.data.slide_name
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    checkpoint_name = os.path.split(args.best_model)[-1]
    np.save(
        os.path.join(
            args.save_path,
            f'{slide_name}_{checkpoint_name}_pred_mask_debug'),
        pred_mask)
    print('done')

    pass


if __name__ == '__main__':
    # pdb.set_trace()
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
