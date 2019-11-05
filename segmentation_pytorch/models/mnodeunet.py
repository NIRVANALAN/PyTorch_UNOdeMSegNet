import torch
import pdb
import torch.nn as nn
import torch.optim as optim

import neuode
from neuode.interface.common import DynamicMap
from neuode.interface.struct import (
	ODEBlockSpec, ConvSpec, SequentialSpec, FFJORDProbDMapSpec,
	ActivationFn, DivSpec,
)
from neuode.dynamics.conv import ConcatSquashConv2d
from neuode.dynamics.composite import build_dmap, SequentialListDMap
from neuode.dynamics.odeblock import ODEBlock
from neuode.dynamics.ffjord_block import FFJORDProbDMap, FFJORDBlock
from neuode.util.util import log_normal_pdf, actfn2nn


class MultiResDownsample(nn.Module):

	def __init__(self, num_res, scale_factor):
		super(MultiResDownsample, self).__init__()
		self.num_res = num_res
		self.downsample = torch.nn.AvgPool2d(scale_factor, stride=scale_factor)

	#         self.downsample = torch.nn.MaxPool2d(scale_factor, stride=scale_factor)

	def forward(self, x):
		y = []
		for _ in range(self.num_res):
			y.append(x)
			x = self.downsample(x)
		return y[::-1]


# class MultiResDMap(DynamicMap):
class MultiResDMap(nn.Module):

	def __init__(self, num_res, dim_latent, scale_factor):
		# super(MultiResDMap, self).__init__()
		super(MultiResDMap, self).__init__()
		# gradient function for each resolution
		# pdb.set_trace()
		MIDDIM = dim_latent * 4
		self.mres_nets = nn.ModuleList()
		for _ in range(num_res):
			self.mres_nets.append(SequentialListDMap([
				ConcatSquashConv2d(2 * dim_latent, MIDDIM, actfn=ActivationFn.TANH),
				ConcatSquashConv2d(MIDDIM, MIDDIM, actfn=ActivationFn.TANH),
				ConcatSquashConv2d(MIDDIM, dim_latent, actfn=ActivationFn.NONE),
			]))

		# upsampling the gradient
		self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

	def forward(self, t, x):
		dxdt = []
		prev_xi = torch.zeros(x[0].shape).to(x[0].device)
		for net, xi in zip(self.mres_nets, x):
			cat_xi = torch.cat([xi, prev_xi], 1)
			dxidt = net(t, cat_xi)
			prev_xi = self.upsample(dxidt)
			dxdt.append(dxidt)
		return dxdt


class MultiResMLP(nn.Module):

	def __init__(self, num_res, dim_in, dim_out):
		super(MultiResMLP, self).__init__()

		# separate mlp for each resolution
		MIDDIM = dim_in * 2
		self.mres_nets = nn.ModuleList()
		for _ in range(num_res):
			self.mres_nets.append(nn.Sequential(
				nn.Conv2d(dim_in, MIDDIM, kernel_size=1, stride=1, padding=0),
				actfn2nn(ActivationFn.RELU),
				nn.Conv2d(MIDDIM, dim_out, kernel_size=1, stride=1, padding=0),
			))

	def forward(self, x):
		logits = []
		for net, xi in zip(self.mres_nets, x):
			logit = net(xi)
			logits.append(logit)
		return logits


class UNOdeMSegNet(nn.Module):

	def __init__(self, dim_in=1, n_classes=8, dim_latent=32, num_res=4, scale_factor=2):
		super(UNOdeMSegNet, self).__init__()
		self.num_res = num_res

		# expand from image to higher-dim feature maps
		self.expand_cfn = nn.Sequential(
			nn.Conv2d(dim_in, dim_latent, kernel_size=3, stride=1, padding=1),
			actfn2nn(ActivationFn.RELU),
			nn.Conv2d(dim_latent, dim_latent, kernel_size=3, stride=1, padding=1),
			actfn2nn(ActivationFn.RELU),
		)

		# downsample to coarser resolution
		self.downsample = MultiResDownsample(num_res, scale_factor)

		# dynamics for each resolution
		mres_cdfn = MultiResDMap(num_res, dim_latent, scale_factor)
		self.cdfn_block = ODEBlock(
			mres_cdfn,
			ODEBlockSpec(use_adjoint=True)
		)

		# classify the multi-res feature
		self.classifier = MultiResMLP(num_res, dim_latent, n_classes)

	def forward(self, x):
		# lift to higher-dim feature and prepend with latent variables
		x = self.expand_cfn(x)
		x = self.downsample(x)
		x = self.cdfn_block(x)
		x = self.classifier(x)

		return x
