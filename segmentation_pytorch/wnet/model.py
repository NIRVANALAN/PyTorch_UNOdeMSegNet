# from ..base import EncoderDecoder
from ..unet.decoder import UnetDecoder
from ..unet.model import Unet
from ..base import Model
from ..encoders import get_encoder
import torch.nn as nn


class Wnet(Model):
	"""# WNet as described in https://arxiv.org/pdf/1711.08506.pdf

	UNet Args:
		encoder_name: name of classification model (without last dense layers) used as feature
			extractor to build segmentation model.
		encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
		decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
		decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
			is used.
		classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
		activation: activation function used in ``.predict(x)`` method for inference.
			One of [``sigmoid``, ``softmax``, callable, None]
		center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)
		attention_type: attention module used in decoder of the model
			One of [``None``, ``scse``]

	Returns:
		``torch.nn.Module``: **WNet**

	.. _Unet:
		https://arxiv.org/pdf/1505.04597

	"""

	def __init__(
			self,
			encoder_name_1='resnet34',
			encoder_name_2='resnet34',
			encoder_weights_1='imagenet',
			encoder_weights_2='imagenet',
			decoder_use_batchnorm=True,
			decoder_channels_1=(256, 128, 64, 32, 16),
			decoder_channels_2=(256, 128, 64, 32, 16),
			classes=8,
			activation='softmax',
			center=False,  # usefull for VGG models
			attention_type=None
	):
		super(Wnet, self).__init__()
		if callable(activation) or activation is None:
			self.activation = activation
		elif activation == 'softmax':
			self.activation = nn.Softmax(dim=1)
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		else:
			raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')
		self.u_enc = Unet(
			encoder_name=encoder_name_1,
			decoder_channels=decoder_channels_1,
			# encoder_weights='imagenet',
			encoder_weights=None,
			activation=activation,
			decoder_use_batchnorm=decoder_use_batchnorm,
			classes=classes)
		self.u_dec = Unet(
			encoder_name=encoder_name_2,
			decoder_channels=decoder_channels_2,
			# encoder_weights='imagenet',
			encoder_weights=None,
			decoder_use_batchnorm=decoder_use_batchnorm,
			activation=activation,
			classes=classes)

		self.name = 'w-{}-{}'.format(encoder_name_1, encoder_name_2)

	def forward(self, x):
		"""Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
		x_mid = self.u_enc(x)
		x_rcnst = self.u_dec(x_mid)
		return {'class': x_mid, 'recovery': x_rcnst}
