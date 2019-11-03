import torch
from torchvision import transforms
from .Microscopy_Data import MicroscopyDataset, TiffDataset
from .DataAugment import WaveletDataAugmemt


def build_train_loader(args):
	# normalize = transforms.Normalize(
	#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

	train_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		normalize,
	])
	train_dataset = MicroscopyDataset(
		args.data.root,
		args.data.train_list,
		args.data.train_img_size,
		output_size=args.data.test_img_size,
		transform=train_transform,
		single_channel_target=args.get(
			'single_channel_target',
			False),
		v_flip=args.data.v_flip,
		h_flip=args.data.h_flip,
		multi_stage=args.get('multi_stage', False))
	if args.data.wavelet:
		train_dataset = WaveletDataAugmemt(train_dataset, 'db1', 3)
	train_batch_size = args.data.train_batch_size

	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=train_batch_size,
		num_workers=args.data.workers)
	return train_loader, train_dataset


def build_val_loader(args):
	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	val_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		normalize,
	])
	val_dataset = MicroscopyDataset(
		args.data.root,
		args.data.test_list,
		args.data.test_img_size,
		args.data.test_img_size,
		transform=val_transform,
		h_flip=False,
		v_flip=False)
	if args.data.wavelet:
		val_dataset = WaveletDataAugmemt(val_dataset, 'db1', 3)
	test_batch_size = args.data.test_batch_size
	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=test_batch_size,
		num_workers=args.data.workers)
	return val_loader, val_dataset


def build_inference_loader(args):
	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	val_transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=3),
		transforms.ToTensor(),
		normalize,
	])
	inference_dataset = TiffDataset(
		args.data.root,
		args.data.slide_name,
		args.data.test_img_size,
		transform=val_transform, overlap_size=args.data.overlap_size, evaluate=args.get('eval', False))
	test_batch_size = args.data.test_batch_size
	shape = inference_dataset.get_img_array_shape()
	if args.data.wavelet:
		inference_dataset = WaveletDataAugmemt(inference_dataset, 'db1', 3)
	inference_loader = torch.utils.data.DataLoader(
		inference_dataset,
		batch_size=test_batch_size,
		shuffle=False,
		num_workers=args.data.workers)
	return inference_loader, inference_dataset, shape

# use same transform for train/val for this example
# trans = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [
#                          0.229, 0.224, 0.225])  # imagenet
# ])

# train_set = MicroscopyDataset(2000, transform=trans)
# val_set = MicroscopyDataset(200, transform=trans)
