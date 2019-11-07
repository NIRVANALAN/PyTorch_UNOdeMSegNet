import os
import sys
from torchvision.models.segmentation import deeplabv3_resnet50

if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
import torch
# import segmentation_models_pytorch as smp
import segmentation_pytorch as smp
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import time
import copy
import yaml
from collections import OrderedDict
import argparse
from easydict import EasyDict
from data import build_val_loader, build_train_loader, build_inference_loader
from segmentation_pytorch.utils.losses import PixelCELoss
from segmentation_pytorch.utils.metrics import MIoUMetric, MPAMetric
from segmentation_pytorch.models import create_model


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
			print(f'Loading class weights from static class list: {class_weight}')
		elif isinstance(args.loss.class_weight, str):
			try:
				class_weight = np.load(args.loss.class_weight)
				class_weight = np.divide(
					1.0, class_weight, where=class_weight != 0)
				class_weight /= np.sum(class_weight)
				print(f'Loading class weights from file {args.loss.class_weight}')
			except OSError as e:
				print(f'Error cannot open class weight file, {e}, exiting')
				exit(-1)

	criterion = PixelCELoss(
		num_classes=num_classes,
		weight=class_weight, multi_stage=num_res)
	metrics = [
		MIoUMetric(num_classes=num_classes, ignore_index=None),
		MPAMetric(num_classes=num_classes,
				  ignore_index=None)  # ignore background
	]

	device = 'cuda'
	lr = args.train.lr
	optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
	# optimizer = torch.optim.Adam([
	# 	{'params': model.decoder.parameters(), 'lr': lr},
	# 	# decrease lr for encoder in order not to permute
	# 	# pre-trained weights with large gradients on training start
	# 	{'params': model.encoder.parameters(), 'lr': lr},
	# ])
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

	num_epochs = args.get('epochs', 200)
	test_slides = args.get('test_slides', ['S1_Helios_1of3_v1270.tiff', 'NA_T4_122117_01.tif',
										   'NA_T4R_122117_19.tif', ])
	test_loader = {}
	for test_slide in test_slides:
		# print(f'inference {test_slide}')
		args.data.slide_name = test_slide
		args.eval = True
		tiff_loader, tiff_dataset, shape = build_inference_loader(args)
		test_loader[test_slide] = tiff_loader
	for i in range(0, num_epochs):
		print('\nEpoch: {}  lr: {}'.format(i, optimizer.param_groups[0]['lr']))
		train_logs = train_epoch.run(train_loader)
		test_valid_log = {'miou': np.array([]), 'mpa': np.array([])}
		for test_slide in test_slides:
			valid_logs = valid_epoch.run(test_loader[test_slide])
			print(f'{test_slide}, miou:{valid_logs["miou"]}, mpa:{valid_logs["mpa"]}')
			test_valid_log['miou'] = np.append(test_valid_log['miou'], valid_logs['miou'])
			test_valid_log['mpa'] = np.append(test_valid_log['mpa'], valid_logs['mpa'])
		exp_lr_scheduler.step()
		# do something (save model, change lr, etc.)
		if max_score < test_valid_log['miou'].mean():
			max_score = test_valid_log['miou'].mean()  # mean
			if not os.path.isdir(args.save_path):
				os.makedirs(args.save_path)
			torch.save(
				model.state_dict(),
				os.path.join(
					args.save_path,
					f'./{max_score}.pth'))
			print(
				f'Model saved: MIOU:{test_valid_log["miou"]}, MPA:{test_valid_log["mpa"]}')



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
