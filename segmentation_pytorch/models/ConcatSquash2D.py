import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import cv2
import tifffile as tiff
from functools import reduce, partial
import itertools
from skimage.measure import block_reduce

from neuode.interface.struct import (
	ODEBlockSpec, FFJORDProbDMapSpec,
	ActivationFn, DivSpec,
)
from neuode.dynamics.conv import ConcatSquashConv2d
from neuode.dynamics.composite import SequentialListDMap
from neuode.dynamics.odeblock import ODEBlock
from neuode.dynamics.ffjord_block import FFJORDProbDMap, FFJORDBlock
from neuode.util.util import actfn2nn


def make_truncate_logpz(dim_latent, dim_reduce=None):
	def truncate_logpz(x):
		# compute prob only for the first dim_latent channels
		x = x[:, :dim_latent, ...]
		x = -x.pow(2)
		if dim_reduce != None:
			x = x.mean(dim_reduce, keepdim=True)
		return x

	return truncate_logpz


class MSegNet(nn.Module):

	def __init__(self, dim_in, dim_out):
		super(MSegNet, self).__init__()
		self.dim_out = dim_out

		# expand from image to higher-dim feature maps
		EXDIM = 64
		self.expand_cfn = nn.Sequential(
			nn.Conv2d(dim_in, EXDIM, kernel_size=3, stride=1, padding=1),
			actfn2nn(ActivationFn.RELU),
			nn.Conv2d(EXDIM, EXDIM, kernel_size=3, stride=1, padding=1),
			actfn2nn(ActivationFn.RELU),
		)

		# convolution dynamics
		MIDDIM = 128
		cdfn = SequentialListDMap([
			ConcatSquashConv2d(dim_out + EXDIM, MIDDIM, actfn=ActivationFn.TANH),
			ConcatSquashConv2d(MIDDIM, MIDDIM, actfn=ActivationFn.TANH),
			ConcatSquashConv2d(MIDDIM, dim_out + EXDIM, actfn=ActivationFn.NONE),
		])
		self.cdfn_block = ODEBlock(
			cdfn,
			ODEBlockSpec(use_adjoint=True)
		)

		# log p(z) using only first d elements
		self.logpdf = make_truncate_logpz(self.dim_out, dim_reduce=None)

	def forward(self, x):
		# lift to higher-dim feature and prepend with latent variables
		x = self.expand_cfn(x)
		z = torch.zeros(x.shape[0], self.dim_out, x.shape[2], x.shape[3])
		x = torch.cat([z, x], 1)

		# step through dynamics
		logpx = self.cdfn_block(x)

		# turn to probability
		logpx = self.logpdf(logpx)

		return logpx
