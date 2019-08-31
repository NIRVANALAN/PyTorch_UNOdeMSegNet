import os
import sys

if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
import torch
import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import time
import copy
import yaml
from collections import OrderedDict
import argparse
from easydict import EasyDict
from data import build_val_loader, build_train_loader
from util.metrics import MIoUMetric, MPAMetric
from util.loss import PixelCELoss
from model import WaveletModel


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
		model = smp.Unet(
			encoder_name=args.model.arch,
			# encoder_weights='imagenet',
			encoder_weights=None,
			activation='softmax',  # whatever, will do .max during metrics calculation
			classes=num_classes)
		pass

	# loss = smp.utils.losses.BCEDiceLoss(eps=1., activation='softmax2d')

	# load class weight
	class_weight = None
	if hasattr(args.loss, 'class_weight'):
		if isinstance(args.loss.class_weight, list):
			print('Loading class weights from static class list')
			class_weight = np.array(args.loss.class_weight)
		elif isinstance(args.loss.class_weight, str):
			print(f'Loading class weights from file {args.loss.class_weight}')
			try:
				class_weight = np.load(args.loss.class_weight)
				class_weight = np.divide(
					1.0, class_weight, where=class_weight != 0)
				class_weight /= np.sum(class_weight)
			except OSError as e:
				print(f'Error cannot open class weight file, {e}, exiting')
				exit(-1)

	criterion = PixelCELoss(
		num_classes=num_classes,
		weight=class_weight)
	metrics = [
		# MIoUMetric(num_classes=num_classes, ignore_index=0),
		MIoUMetric(num_classes=num_classes, ignore_index=None),
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
	args.pixel_ce = True
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
