import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings


class ResNetMultiScaleEncoder(ResNet):

	def __init__(self, dsr=8, Fx_channel=8, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.pretrained = False
		self.dsr = dsr
		self.conv1 = nn.Conv2d(3+Fx_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		del self.fc

	def forward(self, x, F_x=None):
		# merge F_x
		if F_x:  # 8*64*64
			F_x = F.interpolate(F_x, scale_factor=self.dsr)
			x = torch.cat([x, F_x], dim=1)
		x0 = self.conv1(x)
		x0 = self.bn1(x0)
		x0 = self.relu(x0)
		x1 = self.maxpool(x0)
		x1 = self.layer1(x1)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)

		return [x4, x3, x2, x1, x0]

	def load_state_dict(self, state_dict, **kwargs):
		state_dict.pop('fc.bias')
		state_dict.pop('fc.weight')
		super().load_state_dict(state_dict, **kwargs)


resnet_encoders = {
	'resnet18': {
		'encoder': ResNetMultiScaleEncoder,
		'pretrained_settings': pretrained_settings['resnet18'],
		'out_shapes': (512, 256, 128, 64, 64),
		'params': {
			'block': BasicBlock,
			'layers': [2, 2, 2, 2],
		},
	},

	'resnet34': {
		'encoder': ResNetMultiScaleEncoder,
		'pretrained_settings': pretrained_settings['resnet34'],
		'out_shapes': (512, 256, 128, 64, 64),
		'params': {
			'block': BasicBlock,
			'layers': [3, 4, 6, 3],
		},
	},

	'resnet50': {
		'encoder': ResNetMultiScaleEncoder,
		'pretrained_settings': pretrained_settings['resnet50'],
		'out_shapes': (2048, 1024, 512, 256, 64),
		'params': {
			'block': Bottleneck,
			'layers': [3, 4, 6, 3],
		},
	},

	'resnet101': {
		'encoder': ResNetMultiScaleEncoder,
		'pretrained_settings': pretrained_settings['resnet101'],
		'out_shapes': (2048, 1024, 512, 256, 64),
		'params': {
			'block': Bottleneck,
			'layers': [3, 4, 23, 3],
		},
	},

	'resnet152': {
		'encoder': ResNetMultiScaleEncoder,
		'pretrained_settings': pretrained_settings['resnet152'],
		'out_shapes': (2048, 1024, 512, 256, 64),
		'params': {
			'block': Bottleneck,
			'layers': [3, 8, 36, 3],
		},
	},
}
