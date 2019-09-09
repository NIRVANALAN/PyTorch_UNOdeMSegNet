import torch
import torch.nn as nn
from . import functions as F
import numpy as np
import torch.nn.functional as F


class JaccardLoss(nn.Module):
	__name__ = 'jaccard_loss'

	def __init__(self, eps=1e-7, activation='sigmoid'):
		super().__init__()
		self.activation = activation
		self.eps = eps

	def forward(self, y_pr, y_gt):
		return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
	__name__ = 'dice_loss'

	def __init__(self, eps=1e-7, activation='sigmoid'):
		super().__init__()
		self.activation = activation
		self.eps = eps

	def forward(self, y_pr, y_gt):
		return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
	__name__ = 'bce_jaccard_loss'

	def __init__(self, eps=1e-7, activation='sigmoid'):
		super().__init__(eps, activation)
		self.bce = nn.BCEWithLogitsLoss(reduction='mean')

	def forward(self, y_pr, y_gt):
		jaccard = super().forward(y_pr, y_gt)
		bce = self.bce(y_pr, y_gt)
		return jaccard + bce


class BCEDiceLoss(DiceLoss):
	__name__ = 'bce_dice_loss'

	def __init__(self, eps=1e-7, activation='sigmoid'):
		super().__init__(eps, activation)
		self.bce = nn.BCEWithLogitsLoss(reduction='mean')

	def forward(self, y_pr, y_gt):
		dice = super().forward(y_pr, y_gt)
		bce = self.bce(y_pr, y_gt)
		return dice + bce


class PixelCELoss(nn.Module):
	__name__ = 'pixel_ce_loss'

	def __init__(self, normalize_size=False, num_classes=8, weight=None):
		"""

		:type weight: list
		"""
		super().__init__()
		if weight is not None:
			weight = torch.Tensor(weight)
			pass
		self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if normalize_size else 'mean', weight=weight)
		self.normalize_size = normalize_size
		self.num_classes = num_classes

	def forward(self, pred, label):
		# Calculation
		ps_pred = pred
		ps_label = label
		N, C, H, W = ps_pred.size()
		assert ps_label.size() == (N, H, W)

		# shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
		ps_pred = ps_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)

		# shape [N, H, W] -> [NHW]
		ps_label = ps_label.view(N * H * W).detach()
		loss = self.criterion(ps_pred, ps_label)

		return loss


# def __call__(self, pred, label):

def dice_loss(pred, target, smooth=1.):
	pred = pred.contiguous()
	target = target.contiguous()

	intersection = (pred * target).sum(dim=2).sum(dim=2)

	loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

	return loss.mean()


class ReconstructionLoss(nn.Module):
	__name__ = 'reconstruction_loss'

	def __init__(self):
		super().__init__()

	def forward(self, pred, target):
		return torch.sum((pred - target) ** 2)


vertical_sobel = torch.nn.Parameter(torch.from_numpy(np.array([[[[1, 0, -1],
																 [1, 0, -1],
																 [1, 0, -1]]]])).float().cuda(), requires_grad=False)

horizontal_sobel = torch.nn.Parameter(torch.from_numpy(np.array([[[[1, 1, 1],
																   [0, 0, 0],
																   [-1, -1, -1]]]])).float().cuda(),
									  requires_grad=False)


class NCutLoss(nn.Module):
	__name__ = 'normalized_cut_loss'

	# TODO

	def __init__(self):
		super().__init__()

	def gradient_regularization(softmax, device='cuda'):
		vert = torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
		horizontal = torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
		print('vert', torch.sum(vert))
		print('horizontal', torch.sum(horizontal))
		mag = torch.pow(torch.pow(vert, 2) + torch.pow(horizontal, 2), 0.5)
		mean = torch.mean(mag)
		return mean

	def forward(self, w, pred):
		# normalized cut loss, W is the association matrix, P is the K-class assignment
		# ncut = torch.zeros(1)
		# for k in range(pred.size(1)):
		# 	PW = torch.matmul(pred[:, k], w)
		# 	ncut += torch.matmul(PW, pred[:, k]) / PW.sum()
		# return ncut / pred.size(1)
		return self.gradient_regularization(pred)


class WNetLoss(nn.Module):
	__name__ = 'wnet_loss'

	# TODO

	def __init__(self):
		self.ncut_loss = NCutLoss()
		self.rcnt_loss = ReconstructionLoss()
		super().__init__()

	def forward(self, pred, target):
		ncut_loss = self.ncut_loss(pred['class'])
		rcnt_loss = self.rcnt_loss(pred['recovery'], target)
		return ncut_loss + rcnt_loss
