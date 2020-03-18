import torch
from torchvision import transforms
import pdb
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import tifffile as tiff
import time
import copy
import yaml
import argparse
from easydict import EasyDict
import webcolors
from data import build_inference_loader
from segmentation_pytorch.models import create_model
from segmentation_pytorch.utils.metrics import MIoUMetric, MPAMetric
from segmentation_pytorch.utils.losses import PixelCELoss
from segmentation_pytorch.utils.functions import confusion_matrix
import segmentation_pytorch as smp
from tqdm import tqdm
import os
import cv2
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

device = 'cuda'

# def evaluated_checkpoint(args):
#     eval_loader, eval_dataset = build_inference_loader(args)
#     model = torch.load(args.best_model)
#     num_classes = args.model.get('num_classes', 8)
#     criterion = PixelCELoss(num_classes=num_classes)
#     metrics = [
#         MIoUMetric(num_classes=num_classes, ignore_index=None),
#         MPAMetric(num_classes=num_classes, ignore_index=None)
#     ]
#     valid_epoch = smp.utils.train.ValidEpoch(
#         model,
#         loss=criterion,
#         metrics=metrics,
#         device=device,
#         verbose=True,
#     )
#     valid_epoch.run(eval_loader)

category = [
    'bg',
    'PlasmaMembrane',
    'NuclearMembrane',
    'MitochondriaDark',
    'MitochondriaLight',
    'Desmosome',
    'Cytoskeleton',
    'LipidDroplet']

color = [
    'gray',
    'orange',
    'blue',
    'green',
    'azure',
    'red',
    'yellow',
    'pink']


#
def inference_all_tiff(args):
    model = create_model(args)
    print(model)
    model.load_state_dict(torch.load(args.best_model))
    patch_size = args.data.test_img_size
    print(f'patch_size:{patch_size}')
    slides_dir = os.path.join(args.data.root, 'raw')
    slides = os.listdir(slides_dir)
    checkpoint_name = args.best_model.split('/')[-2]
    save_path = os.path.join(args.save_root, str(patch_size), checkpoint_name)
    # evaluate slide
    num_classes = args.data.num_classes
    criterion = PixelCELoss(num_classes=num_classes)
    MIOU = MIoUMetric(num_classes=num_classes, ignore_index=None)
    MPA = MPAMetric(num_classes=num_classes, ignore_index=None)
    metrics = [
        MIOU, MPA
    ]
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=criterion,
        metrics=metrics,
        device=device,
        verbose=True,
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for slide in slides:
        print(f'inference {slide}')
        args.data.slide_name = slide
        tiff_loader, tiff_dataset, shape = build_inference_loader(args)
        pred_mask = np.zeros(shape=shape)
        h, w = shape
        print(f'slide_shape: h:{h} w:{w}')
        h = h // patch_size
        w = w // patch_size
        print(f'patch: h:{h} w:{w}')
        final_label_mask = np.ndarray(
            shape=(*pred_mask.shape, 3), dtype=np.uint8)
        final_diff_mask = final_label_mask.copy()
        color_mask = np.ndarray(
            shape=(patch_size, patch_size, 3), dtype=np.uint8)
        conf_metric = np.ndarray((num_classes, num_classes), dtype=np.int64)
        # inference and visualization
        model.eval()
        activation = nn.Softmax(dim=1)
        for img, (h, w), label in tqdm(tiff_loader):
            label = label.to(device)
            img = img.to(device)

            with torch.no_grad():
                pr_tile_mask = activation(model(img))

            # pr_tile_mask = model.predict(img.to(device))
            miou = MIOU(pr_tile_mask, label)
            mpa = MPA(pr_tile_mask, label)
            pr_tile_mask = pr_tile_mask.argmax(1).cpu().numpy()  # BC*H*W
            conf_matrix = confusion_matrix(
                pr_tile_mask.reshape(-1), label.view(-1), num_classes, False)
            label = label.cpu().detach().numpy()
            conf_metric += conf_matrix
            for i in range(len(h)):
                # print(i)
                x, y = h[i], w[i]
                for j in range(len(category)):
                    # print(category[i])
                    color_mask[pr_tile_mask[i] ==
                               j] = webcolors.name_to_rgb(color[j])

                color_mask = cv2.putText(color_mask, f'{x}_{y}', (10, 60),
                                         cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 1)
                color_mask = cv2.rectangle(color_mask, (0, 0), (patch_size,
                                                                patch_size),
                                           (255, 0, 0), 2)
                diff_mask = color_mask.copy()
                diff_pixel = pr_tile_mask[i] - label[i]
                diff_mask[diff_pixel != 0] = webcolors.name_to_rgb('black')
                color_mask = cv2.putText(color_mask, f'miou:{miou}', (10, 120),
                                         cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 1)
                color_mask = cv2.putText(color_mask, f'mpa:{mpa:.5f}', (10, 160),
                                         cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 1)
                final_label_mask[x *
                                 patch_size:(x +
                                             1) *
                                 patch_size, y *
                                 patch_size:(y +
                                             1) *
                                 patch_size, :] = color_mask
                final_diff_mask[x *
                                patch_size:(x +
                                            1) *
                                patch_size, y *
                                patch_size:(y +
                                            1) *
                                patch_size, :] = diff_mask
        np.set_printoptions(1)
        print(f'conf_metric: \n{conf_metric}')
        slide_name = args.data.slide_name
        colorful_pred_save_path = os.path.join(save_path, slide_name)
        np.savetxt(f'{colorful_pred_save_path}.txt', conf_metric)

        # for i in range(len(category)):
        # 	# print(category[i])
        # 	color_mask[pred_mask == i] = webcolors.name_to_rgb(color[i])

        colorful_pred_mask = Image.fromarray(final_label_mask)
        colorful_pred_mask.save(f'{colorful_pred_save_path}.png', format='png')
        print(f'image saved: {colorful_pred_save_path}')
        final_diff_mask = Image.fromarray(final_diff_mask)
        final_diff_mask.save(
            f'{colorful_pred_save_path}_diff.png', format='png')
        print(f'diff map saved: {colorful_pred_save_path}_diff')
        # visualize_mask(pred_mask, os.path.join(save_path, slide_name), tiff_dataset.get_img_array_shape(), patch_size)

        # metrics
        tiff_dataset.eval = True
        inference_loader = torch.utils.data.DataLoader(
            tiff_dataset,
            batch_size=args.data.test_batch_size,
            shuffle=False,
            num_workers=args.data.workers)
        valid_logs = valid_epoch.run(inference_loader)
        print(
            f'{slide_name}, miou:{valid_logs["miou"]}, mpa:{valid_logs["mpa"]}')
    pass


def visualize_mask(pred_mask, save_path, slide_shape=None, patch_size=512):
    color_mask = np.ndarray(shape=(*pred_mask.shape, 3), dtype=np.uint8)
    # tmp_array = np.array(color_mask)
    for i in range(len(category)):
        # print(category[i])
        color_mask[pred_mask == i] = webcolors.name_to_rgb(color[i])
    h, w = slide_shape
    print(f'slide_shape: h:{h} w:{w}')
    h = h // patch_size
    w = w // patch_size
    print(f'patch: h:{h} w:{w}')

    for y in range(h):
        for x in range(w):
            color_mask = cv2.putText(color_mask, f'{x}_{y}', (x * patch_size + 10, y * patch_size + 60),
                                     cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 3)
            color_mask = cv2.rectangle(color_mask, (x * patch_size, y * patch_size), ((x + 1) * patch_size,
                                                                                      (y + 1) * patch_size),
                                       (255, 0, 0), 2)

    # print(color_mask.dtype)

    colorful_pred_mask = Image.fromarray(color_mask)
    colorful_pred_mask.save(f'{save_path}.png', format='png')
    print(f'image saved: {save_path}')
    # colorful_pred_mask.save(f'64_NAT4R_122117_19_0.57_debug.png', format='png')
    pass


def main(args):
    inference_all_tiff(args)
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
