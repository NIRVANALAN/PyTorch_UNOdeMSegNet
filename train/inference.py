from tqdm import tqdm
from util.loss import PixelCELoss
from util.metrics import MIoUMetric, MPAMetric
from data import build_inference_loader
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
import webcolors
from PIL import Image
from data import build_inference_loader
from util.metrics import MIoUMetric, MPAMetric
from util.loss import PixelCELoss
from tqdm import tqdm

device = 'cuda'


def evaluated_checkpoint(args):
	eval_loader, eval_dataset = build_inference_loader(args)
	model = torch.load(args.best_model)
	num_classes = args.model.get('num_classes', 8)
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
	valid_epoch.run(eval_loader)


#
def inference_all_tiff(args):
	model = torch.load(args.best_model)
	patch_size = args.data.test_img_size
	slides = os.listdir(args.data.root)
	checkpoint_name = args.best_model.split('/')[-2]
	save_path = os.path.join(args.save_root, str(patch_size), checkpoint_name)
	if not os.path.isdir(save_path):
		os.makedirs(save_path)
	for slide in slides:
		print(f'inference {slide}')
		args.data.slide_name = slide
		tiff_loader, tiff_dataset = build_inference_loader(args)
		pred_mask = np.zeros(shape=(tiff_dataset.get_img_array_shape()))
		for img, (h, w) in tqdm(tiff_loader):
			pr_tile_mask = model.predict(img.to(device))
			pr_tile_mask = pr_tile_mask.argmax(1).cpu().numpy()  # BC*H*W
			for i in range(len(h)):
				x, y = h[i], w[i]
				pred_mask[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = pr_tile_mask[i]
		slide_name = args.data.slide_name
		visualize_mask(pred_mask, os.path.join(save_path, slide_name))
	pass


def visualize_mask(pred_mask, save_path):
	category = ['bg', 'PlasmaMembrane', 'NuclearMembrane', 'MitochondriaDark', 'MitochondriaLight', 'Desmosome',
				'Cytoskeleton', 'LipidDroplet']

	color = ['gray', 'orange', 'blue', 'green', 'azure', 'red', 'yellow', 'pink']

	color_mask = np.ndarray(shape=(*pred_mask.shape, 3), dtype=np.uint8)
	# tmp_array = np.array(color_mask)
	for i in range(len(category)):
		# print(category[i])
		color_mask[pred_mask == i] = webcolors.name_to_rgb(color[i])
	# print(color_mask.dtype)

	colorful_pred_mask = Image.fromarray(color_mask)
	colorful_pred_mask.save(f'{save_path}.png', format='png')
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
