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
from data import build_val_loader
from util.metrics import MIoUMetric, MPAMetric
from util.loss import PixelCELoss


def evaluated_checkpoint(args):
	valid_loader, test_dataset = build_val_loader(args)
	model = torch.load(args.best_model)
	num_classes = args.model.get('num_classes', 8)
	criterion = PixelCELoss(num_classes=num_classes)
	device = 'cuda'
	metrics = [
		MIoUMetric(num_classes=num_classes, ignore_index=None),
		MPAMetric(num_classes=num_classes, ignore_index=None)  # ignore background
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
	patch_size = args.data.test_img_size
	device = 'cuda'
	model = torch.load(args.best_model)
	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	val_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		normalize,
	])
	slide_name = args.data.slide_path.split('/')[-1].split('.')[0]
	img_array = (tiff.imread(os.path.join(args.data.root, args.data.slide_path)))
	if img_array.dtype != 'uint16':
		img_array = np.uint16(img_array)
	pred_mask = np.zeros_like(img_array)
	slide_img = Image.fromarray(img_array)
	slide_img = val_transform(slide_img)
	print(slide_img.shape)
	channel, h, w = slide_img.shape  # torch.Size([3, 4096, 6144])
	h = h // patch_size
	w = w // patch_size
	# pdb.set_trace()
	for x in range(h):
		for y in range(w):
			print(x, y)
			tile = slide_img[:, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].to(
				device).unsqueeze(0)
			pr_tile_mask = model.predict(tile).squeeze(0)
			# pdb.set_trace()
			pr_tile_mask = pr_tile_mask.argmax(0)
			pred_mask[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = pr_tile_mask.cpu(

			).numpy()
	if not os.path.isdir(args.save_path):
		os.makedirs(args.save_path)
	checkpoint_name = os.path.split(args.best_model)[-1]
	np.save(os.path.join(args.save_path, f'{slide_name}_{checkpoint_name}_pred_mask'), pred_mask)
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
