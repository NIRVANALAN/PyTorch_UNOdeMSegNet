import segmentation_models_pytorch as smp
import torch
import os
import sys

if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import time
import copy
import yaml
import argparse
from easydict import EasyDict
from data import build_val_loader, build_train_loader
from util.metrics import MIoUMetric, MPAMetric


def main(args):
	num_classes = args.model.get('num_classes', 8)
	model = smp.Unet(
		encoder_name=args.model.arch,
		# encoder_weights='imagenet',
		encoder_weights=None,
		activation='softmax',  # whatever, will do .max during metrics calculation
		classes=num_classes
	)
	loss = smp.utils.losses.BCEDiceLoss(eps=1.)
	metrics = [
		MIoUMetric(num_classes=num_classes),
		MPAMetric(num_classes=num_classes)
	]

	device = 'cuda'
	optimizer = torch.optim.Adam([
		{'params': model.decoder.parameters(), 'lr': 1e-2},

		# decrease lr for encoder in order not to permute
		# pre-trained weights with large gradients on training start
		{'params': model.encoder.parameters(), 'lr': 1e-2},
	])
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.train.lr, momentum=args.train.momentum)
	# dataset
	args.pixel_ce = False
	train_loader = build_train_loader(args)
	valid_loader = build_val_loader(args)

	# create epoch runners
	# it is a simple loop of iterating over dataloader`s samples
	train_epoch = smp.utils.train.TrainEpoch(
		model,
		loss=loss,
		metrics=metrics,
		optimizer=optimizer,
		device=device,
		verbose=True,
	)

	valid_epoch = smp.utils.train.ValidEpoch(
		model,
		loss=loss,
		metrics=metrics,
		device=device,
		verbose=True,
	)

	max_score = 0

	exp_lr_scheduler = lr_scheduler.MultiStepLR(
		optimizer, milestones=args.train.lr_iters, gamma=0.1)

	num_epochs = args.get('epochs', 100)

	for i in range(0, num_epochs):

		print('\nEpoch: {}'.format(i))
		train_logs = train_epoch.run(train_loader)
		valid_logs = valid_epoch.run(valid_loader)

		exp_lr_scheduler.step()

		# do something (save model, change lr, etc.)
		if max_score < valid_logs['miou']:
			max_score = valid_logs['miou']  # mean
			if not os.path.isdir(args.save_path):
				os.makedirs(args.save_path)
			torch.save(model, os.path.join(args.save_path, f'./{max_score}.pth'))
			print(f'Model saved: MIOU:{valid_logs["miou"]}, MPA:{valid_logs["mpa"]}')
	# if i == 25:
	# 	optimizer.param_groups[0]['lr'] = 1e-4
	# 	print('Decrease decoder learning rate to 1e-4!')


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
