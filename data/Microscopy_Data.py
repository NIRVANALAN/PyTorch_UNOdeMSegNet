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
from torchvision import datasets, models, transforms

from util import simulation

object_categories = ['T4', 'T4R', 'S1']
category = ['Cytoskeleton', 'Desmosome', 'LipidDroplet', 'MitochondriaDark', 'MitochondriaLight', 'NuclearMembrane',
            'PlasmaMembrane']


def load_pil(img):
    img = Image.open(img)
    return np.array(img)


def generate_mask(dataset, name, shape=112):
    dataset = osp.join(dataset, 'Mask')
    organelles = os.listdir(dataset)
#     print(organelles)
    shapes = (shape, shape)
    masks = np.zeros((len(category), *shapes)).astype(np.float32)
    print(f'masks shape: {masks.shape}')
    for i in range(len(category)):
        if category[i] in organelles:
            mask = load_pil(osp.join(dataset, category[i], name))
#             print(np.sum(mask))
            if np.sum(mask) != 0:
                print(category[i])
            # mask[mask == 1] = 255
            masks[i] = mask

#     print(f'mask after gen: {masks.shape}')
    return masks


def read_object_labels(file, header=True, shuffle=True):
    images = []
    # num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        for line in f:
            img, cell_type, mask_label = line.split(';')
            cell_label = object_categories.index(cell_type)
            mask_label = eval(mask_label.strip('\n'))
            mask_label = (np.asarray(mask_label)).astype(np.float32)
            mask_label = torch.from_numpy(mask_label)
            images.append((img, cell_label, mask_label))
    if shuffle:
        random.shuffle(images)
    return images


class MicroscopyDataset(Dataset):
    def __init__(self, root, train_list, img_size, transform=None, target_transform=None, crop_size=-1):
        # self.input_images, self.target_masks = simulation.generate_random_data(
        #     192, 192, count=count)
        self.transform = transform
        self.root = root
        self.img_size = img_size
        self.classes = category
        self.transform = transform
        self.crop_size = crop_size
        self.target_transform = target_transform
        self.images = read_object_labels(train_list)

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

    def __getitem__(self, index):
        path, target, mask_target = self.images[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        path_split = path.split('/')
        dataset = '/'.join(path_split[:-2])
        img_name = path_split[-1]
        # path_split[-2] = 'Mask'
        # path_split.insert(-1,'organelle')

        # mask
        mask = generate_mask(dataset, img_name)

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
