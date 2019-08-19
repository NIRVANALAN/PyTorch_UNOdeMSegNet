from collections import defaultdict
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch
from torch import autograd
import time
import pdb
import copy
import os
import gc


# def calc_loss(pred, target, metrics, bce_weight=0.5):
# 	bce = F.binary_cross_entropy_with_logits(pred, target)
#
# 	pred = torch.sigmoid(pred)
# 	dice = dice_loss(pred, target)
#
# 	loss = bce * bce_weight + dice * (1 - bce_weight)
#
# 	metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
# 	metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
# 	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
#
# 	return loss

def calc_seg_acc(outputs, target):
	# pdb.set_trace()
	pred = torch.argmax(outputs, 1)  # reduce channel dim
	N, H, W = target.size()
	acc = (pred == target).sum().to(torch.float) / (N * H * W) * 100
	return acc.data.cpu().numpy()


# print(f'segmentation acc: {acc * 100:5.2f}%')


def print_metrics(metrics, epoch_samples, phase):
	outputs = []
	for k in metrics.keys():
		outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

	print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, criterion, num_epochs=25, args=None):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 1e10
	best_acc = 1e-10

	t = range(num_epochs)

	with autograd.detect_anomaly():
		for _ in t:
			total_loss = 0
			print('Epoch {}/{}'.format(_, num_epochs - 1))
			# print('-' * 10)

			since = time.time()

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					scheduler.step()
					for param_group in optimizer.param_groups:
						print("LR", param_group['lr'])
					model.train()  # Set model to training mode
				else:
					model.eval()  # Set model to evaluate mode

				metrics = defaultdict(float)
				epoch_samples = 0
				gc.collect()

				for inputs, labels in tqdm(dataloaders[phase]):
					inputs = inputs.to(device)
					labels = labels.to(device)  # BC * channel

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						# pdb.set_trace()
						outputs = model(inputs)
						loss = criterion(outputs, labels)
						metrics['seg_acc'] += calc_seg_acc(outputs, labels) * labels.size(0)
						metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
						# total_loss += loss

						# loss = calc_loss(outputs, labels, metrics)

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							optimizer.step()

					# statistics
					epoch_samples += inputs.size(0)
				# print(f"Total loss in epoch {_ + 1} : {total_loss / epoch_samples}")

				print_metrics(metrics, epoch_samples, phase)
				epoch_loss = metrics['loss'] / epoch_samples
				epoch_acc = metrics['seg_acc'] / epoch_samples
				gc.collect()

				# deep copy the model
				if phase == 'val' and epoch_loss < best_loss:
					print("saving best model")
					best_loss = epoch_loss
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())

					checkpoint = {'model': model,
								  'state_dict': model.state_dict(),
								  'optimizer': optimizer.state_dict()}
					save_path = args.save_path
					if not os.path.isdir(save_path):
						os.mkdir(save_path)
					torch.save(checkpoint, os.path.join(save_path, f"epoch{_}_acc{best_acc}.pth"))

		time_elapsed = time.time() - since
		print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	print('Best val loss: {:4f}, acc: {:.4f}%'.format(best_loss, best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)

	# checkpoint = {'model': model,
	# 			  'state_dict': model.state_dict(),
	# 			  'optimizer': optimizer.state_dict()}
	# save_path = args.save_path
	# if not os.path.isdir(save_path):
	# 	os.mkdir(save_path)
	# torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth'))
	return model
