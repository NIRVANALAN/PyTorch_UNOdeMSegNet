import io
import os
import os.path as osp
import random
import sys
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import vflip, hflip, rotate
from torchvision import datasets, models, transforms

from util import simulation

object_categories = ['T4', 'T4R', 'S1']
category = ['Cytoskeleton', 'Desmosome', 'LipidDroplet', 'MitochondriaDark', 'MitochondriaLight', 'NuclearMembrane',
			'PlasmaMembrane']


def load_pil(img, shape=None):
	img = Image.open(img)
	if shape:
		img = img.resize((shape, shape), Image.BILINEAR)
	return np.array(img)


def generate_mask(dataset, img_name, shape=484, single_channel_target=False, include_bg=True):
	"""

	:param include_bg: whether include BG mask as a class
	:param single_channel_target: return stacked or not
	:param dataset: 256_dataset/256/T4R/NA_T4R_122117_19/NA_T4R_122117_19
	:param img_name: 11_5.png
	:param shape: 256
	:return: np.array -> (num_class, shape, shape)
	"""
	# dataset = osp.join(dataset, 'Mask')
	all_slides = os.listdir('/'.join(dataset.split('/')[:-1]))
	raw_slide_name = dataset.split('/')[-1]

	shapes = (shape, shape)
	if single_channel_target:
		masks = np.zeros((len(category) + 1, *shapes)).astype(np.int_)  # long
		# print(f'masks shape: {masks.shape}')
		for i in range(len(category)):
			if f'{raw_slide_name}_{category[i]}' in all_slides:
				mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
				masks[i + 1] = mask
		target_mask = np.argmax(masks, 0)  # bg->0
	elif include_bg:
		masks = np.zeros((len(category), *shapes)).astype(np.float_)  # long
		# print(f'masks shape: {masks.shape}')
		for i in range(len(category)):
			if f'{raw_slide_name}_{category[i]}' in all_slides:
				mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
				masks[i] = mask
		bg_mask = np.expand_dims(1 - np.clip(np.sum(masks, 0), 0, 1), axis=0)  # 1 * H * W. some pixels have
		# multi-label, visualization is needed
		target_mask = np.concatenate((bg_mask, masks), axis=0)  # 8*H*W, zero dimension as BG
	else:
		masks = np.zeros((len(category), *shapes)).astype(np.float_)  # long
		# print(f'masks shape: {masks.shape}')
		for i in range(len(category)):
			if f'{raw_slide_name}_{category[i]}' in all_slides:
				mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
				masks[i] = mask
		target_mask = masks
	#     print(f'mask after gen: {masks.shape}')
	return target_mask.astype(np.float32)


def read_object_labels(file, header=True, shuffle=True):
	images = []
	# num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		for line in f:
			line_split = line.split(';')
			if line_split.__len__() == 3:
				img, cell_type, mask_label = line_split
				cell_label = object_categories.index(cell_type)
				mask_label = eval(mask_label.strip('\n'))
				mask_label = (np.asarray(mask_label)).astype(np.float32)
				mask_label = torch.from_numpy(mask_label)
				images.append((img, cell_label, mask_label))
			else:
				img, cell_type = line_split
				cell_label = object_categories.index(cell_type.strip('\n'))
				images.append((img, cell_label))
	if shuffle:
		random.shuffle(images)
	return images


class MicroscopyDataset(Dataset):
	def __init__(self, root, train_list, img_size, output_size, transform=None, target_transform=None, crop_size=-1,
				 h_flip=True,
				 v_flip=True, single_channel_target=False, include_bg=True):
		self.transform = transform
		self.root = root
		self.h_flip = h_flip
		self.v_flip = v_flip
		self.img_size = img_size
		self.output_size = output_size
		self.classes = category
		self.transform = transform
		self.crop_size = crop_size
		self.target_transform = target_transform
		self.images = read_object_labels(train_list)
		self.single_channel_target = single_channel_target
		self.include_bg = include_bg

	def __len__(self):
		return len(self.images)

	def get_number_classes(self):
		return len(self.classes)

	def __getitem__(self, index):
		path, target = self.images[index]
		img = Image.open(os.path.join(self.root, path)).convert('LA')  # gray scale
		# img = Image.open(os.path.join(self.root, path)).convert('RGB')  # gray scale
		if img.size[0] != self.img_size:
			img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
		path_split = path.split('/')
		# if target == 'S1':  # S1 stack hierarchy
		# 	mask_dataset = '/'.join(path_split[:-2])
		# 	pass
		# else:
		mask_dataset = '/'.join(path_split[:-1])
		mask_dataset = os.path.join(self.root, mask_dataset)
		img_name = path_split[-1]
		mask = generate_mask(mask_dataset, img_name, shape=self.output_size,
							 single_channel_target=self.single_channel_target, include_bg=self.include_bg)

		if self.h_flip:
			if np.random.random() < 0.5:
				img = hflip(img)
				mask = np.flip(mask, 1).copy()

		if self.v_flip:
			if np.random.random() < 0.5:
				img = vflip(img)
				mask = np.flip(mask, 0).copy()

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return [img, mask]


# def __getitem__(self, idx):
#     image = self.input_images[idx]
#     mask = self.target_masks[idx]
#     if self.transform:
#         image = self.transform(image)

#     return [image, mask]


# class MicroscopyClassification(data.Dataset):
#     def __init__(self, root, train_list, img_size, transform=None, target_transform=None, crop_size=-1):
#         self.root = root
#         self.img_size = img_size
#         # self.path_images = os.path.join(root, 'JPEGImage')
#         # self.path_annotation = os.path.join(root, 'Annotation')

#         self.transform = transform
#         self.crop_size = crop_size
#         self.target_transform = target_transform

#         self.classes = object_categories
#         self.images = read_object_labels(train_list)


#         print('[dataset] Microscopy classification number of classes=%d  number of images=%d' % (
#             len(self.classes), len(self.images)))

#     def __getitem__(self, index):
#         path, target, mask_target = self.images[index]
#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#         img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
#         # if self.crop_size > 0:
#         # 	start_w = int((self.img_size - self.crop_size) * np.random.random())
#         # 	start_h = int((self.img_size - self.crop_size) * np.random.random())
#         # 	img = img.crop((start_w, start_h, start_w +
#         # 					self.crop_size, start_h + self.crop_size))

#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return (img, path), target

#     def __len__(self):
#         return len(self.images)

#     def get_number_classes(self):
#         return len(self.classes)


def reverse_transform(inp):
	inp = inp.numpy().transpose((1, 2, 0))
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	inp = std * inp + mean
	inp = np.clip(inp, 0, 1)
	inp = (inp * 255).astype(np.uint8)

	return inp
