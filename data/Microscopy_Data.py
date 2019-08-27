import io
import tifffile as tiff
import os
import os.path as osp
import cv2
import pdb
import random
import sys
import traceback
from typing import Any, Union, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from numpy.core._multiarray_umath import ndarray
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import vflip, hflip, rotate
from torchvision import datasets, models, transforms

from util import simulation

object_categories = ['T4', 'T4R', 'S1']


# category = ['PlasmaMembrane', 'NuclearMembrane', 'MitochondriaDark', 'MitochondriaLight', 'Desmosome', 'Cytoskeleton',
#             'LipidDroplet']


def load_pil(img, shape=None):
	img = Image.open(img)
	if shape:
		img = img.resize((shape, shape), Image.BILINEAR)
	return np.array(img)


def generate_mask(mask_dir, img_name, shape=512, num_classes=8):
	"""

	:param num_classes: num of classes
	:param mask_dir: root_dir/256_dataset/256/T4R/NA_T4R_122117_19/NA_T4R_122117_19_label
	:param img_name: 11_5.png
	:param shape: 256
	:return: np.array -> (num_class, shape, shape)
	"""
	# dataset = osp.join(dataset, 'Mask')
	# all_slides = os.listdir('/'.join(dataset.split('/')[:-1]))
	# label_dir = f'{dataset}_label'
	# raw_slide_name = dataset.split('/')[-1]
	label_name = img_name + '.npy'
	# print(label_name)
	mask: np.ndarray = np.load(os.path.join(mask_dir, label_name))
	# print(f'load {mask_dir}_{label_name}')
	if mask.shape[0] != shape:
		mask = cv2.resize(mask, (shape, shape), cv2.INTER_NEAREST)
	# print(mask.shape)
	assert np.ndim(mask) == 2 or np.ndim(mask) == 3
	if np.ndim(mask) == 2:
		one_hot_mask = np.ndarray((num_classes, *mask.shape))
		one_hot_mask.fill(0)
		for i in range(mask.shape[0]):
			for j in range(mask.shape[1]):
				one_hot_mask[mask[i][j]][i][j] = 1
		target = one_hot_mask
	else:
		target = mask

	# if single_channel_target:
	# 	masks = np.zeros((len(category) + 1, *shapes)).astype(np.int_)  # long
	# 	# print(f'masks shape: {masks.shape}')
	# 	for i in range(len(category)):
	# 		if f'{raw_slide_name}_{category[i]}' in all_slides:
	# 			mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
	# 			masks[i + 1] = mask
	# 	target_mask = np.argmax(masks, 0)  # bg->0
	# elif include_bg:
	# 	masks = np.zeros((len(category), *shapes)).astype(np.float_)  # long
	# 	# print(f'masks shape: {masks.shape}')
	# 	for i in range(len(category)):
	# 		if f'{raw_slide_name}_{category[i]}' in all_slides:
	# 			mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
	# 			masks[i] = mask
	# 	bg_mask = np.expand_dims(1 - np.clip(np.sum(masks, 0), 0, 1), axis=0)  # 1 * H * W. some pixels have
	# 	# multi-label, visualization is needed
	# 	target_mask = np.concatenate((bg_mask, masks), axis=0)  # 8*H*W, zero dimension as BG
	# else:
	# 	masks = np.zeros((len(category), *shapes)).astype(np.float_)  # long
	# 	# print(f'masks shape: {masks.shape}')
	# 	for i in range(len(category)):
	# 		if f'{raw_slide_name}_{category[i]}' in all_slides:
	# 			mask = load_pil(osp.join(f'{dataset}_{category[i]}', img_name), shape=shape)
	# 			masks[i] = mask
	# 	target_mask = masks
	# #     print(f'mask after gen: {masks.shape}')
	return target.astype(np.float32)


def read_object_labels(file, header=True, shuffle=True):
	images = []
	# num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		for line in f:
			line_split = line.split(';')
			if line_split.__len__() == 3:
				img, segment_label, cell_type = line_split
				cell_label = object_categories.index(cell_type.strip('\n'))
				images.append((img, segment_label, cell_label))
			else:
				img, cell_type = line_split
				cell_label = object_categories.index(cell_type.strip('\n'))
				images.append((img, cell_label))
	if shuffle:
		random.shuffle(images)
	return images


class MicroscopyDataset(Dataset):
	def __init__(self, root, train_list, img_size, output_size, transform=None, target_transform=None, crop_size=-1,
				 h_flip=False,
				 v_flip=False, single_channel_target=False, include_bg=True):
		self.transform = transform
		self.root = root
		self.h_flip = h_flip
		self.v_flip = v_flip
		self.img_size = img_size
		self.output_size = output_size
		# self.classes = category
		self.transform = transform
		self.crop_size = crop_size
		self.target_transform = target_transform
		self.images = read_object_labels(train_list)
		self.single_channel_target = single_channel_target

	def __len__(self):
		return len(self.images)

	def get_number_classes(self):
		return len(self.classes)

	def __getitem__(self, index):
		path, segment_label_path, cell_type = self.images[index]
		if 'tif' in path:
			img = tiff.imread(os.path.join(self.root, path))
		else:
			img = np.load(os.path.join(self.root, path))
		mask = np.load(osp.join(self.root, segment_label_path)).copy()  # 2d mask
		# print(mask.shape)
		# img = Image.open(os.path.join(self.root, path)).convert('RGB')  # gray scale
		if img.shape[0] != self.img_size:
			img = cv2.resize(img, (self.img_size, self.img_size), cv2.INTER_NEAREST)
		if mask.shape[0] != self.img_size:
			# pdb.set_trace()
			mask = cv2.resize(mask, (self.img_size, self.img_size), cv2.INTER_NEAREST)
		# pdb.set_trace()
		img = Image.fromarray(img)
		# path_split = path.split('/')
		# mask_dataset = '/'.join(path_split[:-1])
		# mask_dataset = os.path.join(self.root, mask_dataset + '_label')
		# img_name = path_split[-1]
		if self.v_flip:
			if np.random.random() < 0.5:
				img = vflip(img)
				mask = np.flip(mask, 0).copy()
		if self.h_flip:
			if np.random.random() < 0.5:
				img = hflip(img)
				mask = np.flip(mask, 1).copy()

		if self.transform is not None:
			img = self.transform(img)
		return [img, mask]


class TiffDataset(Dataset):
	def __init__(self, root, slide_name, patch_size, transform=None, target_transform=None):
		self.transform = transform
		self.root = root
		self.patch_size = patch_size
		self.transform = transform
		self.target_transform = target_transform
		self.img_array = (tiff.imread(os.path.join(root, slide_name)))
		if self.img_array.dtype != 'uint16':
			self.img_array = np.uint16(self.img_array)
		self.slide_img = Image.fromarray(self.img_array)
		if self.transform is not None:
			self.slide_img = self.transform(self.slide_img)
		channel, h, w = self.slide_img.shape  # torch.Size([3, 4096, 6144])
		self.h = h // patch_size
		self.w = w // patch_size
		self.images = []
		for x in range(self.h):
			for y in range(self.w):
				# print(x, y)
				self.images.append(self.slide_img[:, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) *
																									patch_size])

	def __len__(self):
		return len(self.images)

	def get_img_array_shape(self):
		return self.img_array.shape

	def __getitem__(self, index):
		x = index // self.w
		y = index % self.w
		img = self.images[index]
		return [img, (x, y)]


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
