import torch.nn as nn
import warnings
import numpy as np
import torch
from .functions import *


# warnings.filterwarnings("error")

#
# class IoUMetric(nn.Module):
# 	__name__ = 'iou'
#
# 	def __init__(self, eps=1e-7, threshold=0.5, activation='sigmoid'):
# 		super().__init__()
# 		self.activation = activation
# 		self.eps = eps
# 		self.threshold = threshold
#
# 	def forward(self, y_pr, y_gt):
# 		return F.iou(y_pr, y_gt, self.eps, self.threshold, self.activation)
#
# #
# class MPAMetric(nn.Module):
# 	__name__ = 'mpa'
#
# 	def __init__(self, activate='one_hot', ignore_index=None):
# 		super().__init__()
# 		self.ignore_index = ignore_index
# 		self.activate = activate
#
# 	def forward(self, pr, gt):
# 		if pr.dim() == 4 and self.activate == 'one_hot':  # N*C*H*W
# 			one_hot_pr = torch.zeros_like(pr)
# 			idx = pr.argmax(1)
# 			one_hot_pr[np.arange(pr.shape[0])[:, None, None, None], idx, np.arange(pr.shape[2])[None, None, :,
# 																		 None], np.arange(pr.shape[3])[None, None, None,
# 																				:]] = 1
# 			return mpa(one_hot_pr, gt, ignore_index=self.ignore_index)
# 		return mpa(pr, gt, ignore_index=self.ignore_index)


#
#
# class MeanIoUMetric(nn.Module):
# 	__name__ = 'mean_iou'
#
# 	def __init__(self, activate='one_hot', ignore_index=None):
# 		super().__init__()
# 		self.ignore_index = ignore_index
# 		self.activate = activate
#
# 	def forward(self, pr, gt):
# 		if pr.dim() == 4 and self.activate == 'one_hot':  # N*C*H*W
# 			one_hot_pr = torch.zeros_like(pr)
# 			# pdb.set_trace()
# 			idx = pr.argmax(1)
#
# 			# for bc in range(pr.shape[0]):
# 			# 	for h in range(pr.shape[2]):
# 			# 		for w in range(pr.shape[3]):
# 			# 			one_hot_pr[bc][idx[bc][h][w]][h][w] = 1
# 			# one_hot_mask[mask[i][j]][i][j] = 1
#
# 			one_hot_pr[np.arange(pr.shape[0])[:, None, None, None], idx, np.arange(pr.shape[2])[None, None, :,
# 																		 None], np.arange(pr.shape[3])[None, None, None,
# 																				:]] = 1
# 			pdb.set_trace()
# 			return miou(one_hot_pr, gt, ignore_index=self.ignore_index)
# 		return miou(pr, gt, ignore_index=self.ignore_index)


class Metric(object):
	"""Base class for all metrics.
	From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
	"""

	def reset(self):
		pass

	def add(self, predicted, target):
		pass

	def value(self):
		pass


class ConfusionMatrix(Metric):
	"""Constructs a confusion matrix for a multi-class classification problems.
	Does not support multi-label, multi-class problems.
	Keyword arguments:
	- num_classes (int): number of classes in the classification problem.
	- normalized (boolean, optional): Determines whether or not the confusion
	matrix is normalized or not. Default: False.
	Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
	"""

	def __init__(self, num_classes, normalized=False):
		super().__init__()

		self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
		self.normalized = normalized
		self.num_classes = num_classes
		self.reset()

	def reset(self):
		self.conf.fill(0)

	def add(self, predicted, target):
		self.conf += confusion_matrix(predicted, target, self.num_classes)

	def value(self):
		"""
		Returns:
			Confustion matrix of K rows and K columns, where rows corresponds
			to ground-truth targets and columns corresponds to predicted
			targets.
		"""
		if self.normalized:
			conf = self.conf.astype(np.float32)
			return conf / conf.sum(1).clip(min=1e-12)[:, None]
		else:
			return self.conf


class SegmentationMetric(Metric):
	"""Computes the intersection over union (IoU) and PA per class and corresponding
	mean (mIoU, mPA).
	Intersection over union (IoU) is a common evaluation metric for semantic
	segmentation. The predictions are first accumulated in a confusion matrix
	and the IoU is computed from it as follows:
		IoU = true_positive / (true_positive + false_positive + false_negative).
	Pixel accuracy:
		PA = (tp+tn) / (tp+tn+fp+fn)
	Keyword arguments:
	- num_classes (int): number of classes in the classification problem
	- normalized (boolean, optional): Determines whether or not the confusion
	matrix is normalized or not. Default: False.
	- ignore_index (int or iterable, optional): Index of the classes to ignore
	when computing the IoU. Can be an int, or any iterable of ints.
	"""

	def __init__(self, num_classes, normalized=False, ignore_index=None):
		super().__init__()
		self.conf_metric = ConfusionMatrix(num_classes, normalized)

		if ignore_index is None:
			self.ignore_index = None
		elif isinstance(ignore_index, int):
			self.ignore_index = (ignore_index,)
		else:
			try:
				self.ignore_index = tuple(ignore_index)
			except TypeError:
				raise ValueError("'ignore_index' must be an int or iterable")

	def reset(self):
		self.conf_metric.reset()

	def add(self, predicted, target):
		"""Adds the predicted and target pair to the IoU metric.
		Keyword arguments:
		- predicted (Tensor): Can be a (N, K, H, W) tensor of
		predicted scores obtained from the model for N examples and K classes,
		or (N, H, W) tensor of integer values between 0 and K-1.
		- target (Tensor): Can be a (N, K, H, W) tensor of
		target scores for N examples and K classes, or (N, H, W) tensor of
		integer values between 0 and K-1.
		"""
		# Dimensions check
		assert predicted.size(0) == target.size(0), \
			'number of targets and predicted outputs do not match'
		assert predicted.dim() == 3 or predicted.dim() == 4, \
			"predictions must be of dimension (N, H, W) or (N, K, H, W)"
		assert target.dim() == 3 or target.dim() == 4, \
			"targets must be of dimension (N, H, W) or (N, K, H, W)"

		# If the tensor is in categorical format convert it to integer format
		if predicted.dim() == 4:
			_, predicted = predicted.max(1)
		if target.dim() == 4:
			_, target = target.max(1)

		self.conf_metric.add(predicted.view(-1), target.view(-1))

	def value(self):
		"""Computes the IoU and mean IoU.
		The mean computation ignores NaN elements of the IoU array.
		Returns:
			Tuple: (IoU, mIoU, mPA). The first output is the per class IoU,
			for K classes it's numpy.ndarray with K elements. The second output,
			is the mean IoU. The third output is the mean Pixel Accuracy
		"""
		conf_matrix = self.conf_metric.value()
		if self.ignore_index is not None:
			for index in self.ignore_index:
				conf_matrix[:, self.ignore_index] = 0
				conf_matrix[self.ignore_index, :] = 0
		true_positive = np.diag(conf_matrix)
		false_positive = np.sum(conf_matrix, 0) - true_positive
		false_negative = np.sum(conf_matrix, 1) - true_positive
		true_negative = np.sum(conf_matrix) - true_positive - false_negative - false_positive

		# Just in case we get a division by 0, ignore/hide the error
		with np.errstate(divide='ignore', invalid='ignore'):
			iou = true_positive / (true_positive + false_positive + false_negative)
			pa = (true_positive + true_negative) / np.sum(conf_matrix)

		return iou, np.nanmean(iou), pa, np.nanmean(pa)


class MIoUMetric(nn.Module):
	__name__ = 'miou'

	def __init__(self, num_classes, ignore_index=None, normalized=False, eps=1e-7, ):
		super().__init__()
		self.eps = eps
		self.num_classes = num_classes
		self.normalized = normalized
		if ignore_index is None:
			self.ignore_index = None
		elif isinstance(ignore_index, int):
			self.ignore_index = (ignore_index,)
		else:
			try:
				self.ignore_index = tuple(ignore_index)
			except TypeError:
				raise ValueError("'ignore_index' must be an int or iterable")

	def forward(self, y_pr, y_gt):

		"""
		:param y_pr: Can be a (N, K, H, W) tensor of
		predicted scores obtained from the model for N examples and K classes,
		or (N, H, W) tensor of integer values between 0 and K-1.
		:param y_gt: Can be a (N, K, H, W) tensor of
		target scores for N examples and K classes, or (N, H, W) tensor of
		integer values between 0 and K-1.
		:return: Mean IoU
		"""
		# Dimensions check
		assert y_pr.size(0) == y_gt.size(0), \
			'number of targets and predicted outputs do not match'
		assert y_pr.dim() == 3 or y_pr.dim() == 4, \
			"predictions must be of dimension (N, H, W) or (N, K, H, W)"
		assert y_gt.dim() == 3 or y_gt.dim() == 4, \
			"targets must be of dimension (N, H, W) or (N, K, H, W)"

		# If the tensor is in categorical format convert it to integer format
		if y_pr.dim() == 4:
			_, y_pr = y_pr.max(1)
		if y_gt.dim() == 4:
			_, y_gt = y_gt.max(1)

		# pdb.set_trace()
		conf_matrix = confusion_matrix(y_pr.view(-1), y_gt.view(-1), self.num_classes, self.normalized)
		if self.ignore_index is not None:
			for index in self.ignore_index:
				conf_matrix[:, index] = 0
				conf_matrix[index, :] = 0
		true_positive = np.diag(conf_matrix)
		false_positive = np.sum(conf_matrix, 0) - true_positive
		false_negative = np.sum(conf_matrix, 1) - true_positive
		batch_gt = np.sum(conf_matrix, 1)
		class_exist = np.nonzero(batch_gt)
		# print(class_exist)

		# true_negative = np.sum(conf_matrix) - true_positive - false_negative - false_positive

		# Just in case we get a division by 0, ignore/hide the error
		# with np.errstate(divide='ignore', invalid='ignore'):
		iou_score = (true_positive + self.eps) / (true_positive + false_positive + false_negative + self.eps)
		iou_score = iou_score[class_exist]
		# pdb.set_trace()
		# with warnings.catch_warnings() as w:
		# Cause all warnings to always be triggered.
		# warnings.simplefilter("error", category=RuntimeError)
		warnings.filterwarnings('error')
		try:
			iou_score = torch.tensor(np.nanmean(iou_score))
		except RuntimeWarning:
			pdb.set_trace()
		# pa = (true_positive + true_negative) / np.sum(conf_matrix)
		# 	pdb.set_trace()
		return iou_score


class MPAMetric(nn.Module):
	__name__ = 'mpa'

	def __init__(self, num_classes, ignore_index=None, normalized=False):
		super().__init__()
		self.num_classes = num_classes
		self.normalized = normalized
		if ignore_index is None:
			self.ignore_index = None
		elif isinstance(ignore_index, int):
			self.ignore_index = (ignore_index,)
		else:
			try:
				self.ignore_index = tuple(ignore_index)
			except TypeError:
				raise ValueError("'ignore_index' must be an int or iterable")

	def forward(self, y_pr, y_gt):

		"""
		:param y_pr: Can be a (N, K, H, W) tensor of
		predicted scores obtained from the model for N examples and K classes,
		or (N, H, W) tensor of integer values between 0 and K-1.
		:param y_gt: Can be a (N, K, H, W) tensor of
		target scores for N examples and K classes, or (N, H, W) tensor of
		integer values between 0 and K-1.
		:return: Mean IoU
		"""
		# Dimensions check
		assert y_pr.size(0) == y_gt.size(0), \
			'number of targets and predicted outputs do not match'
		assert y_pr.dim() == 3 or y_pr.dim() == 4, \
			"predictions must be of dimension (N, H, W) or (N, K, H, W)"
		assert y_gt.dim() == 3 or y_gt.dim() == 4, \
			"targets must be of dimension (N, H, W) or (N, K, H, W)"

		# If the tensor is in categorical format convert it to integer format
		if y_pr.dim() == 4:
			_, y_pr = y_pr.max(1)
		if y_gt.dim() == 4:
			_, y_gt = y_gt.max(1)

		conf_matrix = confusion_matrix(y_pr.view(-1), y_gt.view(-1), self.num_classes, self.normalized)

		if self.ignore_index is not None:
			for index in self.ignore_index:
				conf_matrix[:, index] = 0
				conf_matrix[index, :] = 0
		true_positive = np.diag(conf_matrix)
		false_positive = np.sum(conf_matrix, 0) - true_positive
		false_negative = np.sum(conf_matrix, 1) - true_positive
		true_negative = np.sum(conf_matrix) - true_positive - false_negative - false_positive
		batch_gt = np.sum(conf_matrix, 1)
		class_exist = np.nonzero(batch_gt)

		# Just in case we get a division by 0, ignore/hide the error
		# with np.errstate(divide='ignore', invalid='ignore'):

		with warnings.catch_warnings() as w:
			warnings.simplefilter("error", category=RuntimeError)
			pa = (true_positive + true_negative) / np.sum(conf_matrix)
			pa = pa[class_exist]
			try:
				return torch.tensor(np.nanmean(pa))
			except:
				pdb.set_trace()
