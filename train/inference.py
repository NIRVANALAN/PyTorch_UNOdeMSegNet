from tqdm import tqdm
import os
import cv2
import sys

if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
from segmentation_pytorch.utils.losses import PixelCELoss
from segmentation_pytorch.utils.metrics import MIoUMetric, MPAMetric
from data import build_inference_loader
import webcolors
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
import matplotlib.pyplot as plt
from PIL import Image
import pdb
from torchvision import transforms
import segmentation_models_pytorch as smp
from model import WaveletModel
import torch

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


#
def inference_all_tiff(args):
	model = torch.load(args.best_model)
	patch_size = args.data.test_img_size
	print(f'patch_size:{patch_size}')
	slides_dir = os.path.join(args.data.root, 'raw')
	slides = os.listdir(slides_dir)
	checkpoint_name = args.best_model.split('/')[-2]
	save_path = os.path.join(args.save_root, str(patch_size), checkpoint_name)
	# evaluate slide
	num_classes = args.data.num_classes
	criterion = PixelCELoss(num_classes=num_classes)
	metrics = [
		MIoUMetric(num_classes=num_classes, ignore_index=None),
		MPAMetric(num_classes=num_classes, ignore_index=None)
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
		pred_mask_no = pred_mask.copy()
		# inference and visualization
		for img, (h, w) in tqdm(tiff_loader):
			pr_tile_mask = model.predict(img.to(device))
			pr_tile_mask = pr_tile_mask.argmax(1).cpu().numpy()  # BC*H*W
			for i in range(len(h)):
				x, y = h[i], w[i]
				pred_mask[x *
						  patch_size:(x +
									  1) *
									 patch_size, y *
												 patch_size:(y +
															 1) *
															patch_size] = pr_tile_mask[i]
		slide_name = args.data.slide_name
		visualize_mask(pred_mask, os.path.join(save_path, slide_name), tiff_dataset.get_img_array_shape(), patch_size)
		# metrics
		tiff_dataset.eval = True
		inference_loader = torch.utils.data.DataLoader(
			tiff_dataset,
			batch_size=args.data.test_batch_size,
			shuffle=False,
			num_workers=args.data.workers)
		valid_logs = valid_epoch.run(inference_loader)
		print(f'{slide_name}, miou:{valid_logs["miou"]}, mpa:{valid_logs["mpa"]}')
	pass


def visualize_mask(pred_mask, save_path, slide_shape=None, patch_size=512):
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

	color_mask = np.ndarray(shape=(*pred_mask.shape, 3), dtype=np.uint8)
	# tmp_array = np.array(color_mask)
	for i in range(len(category)):
		# print(category[i])
		color_mask[pred_mask == i] = webcolors.name_to_rgb(color[i])
	h, w = slide_shape
	h = h // patch_size
	w = w // patch_size

	for x in range(h):
		for y in range(w):
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
