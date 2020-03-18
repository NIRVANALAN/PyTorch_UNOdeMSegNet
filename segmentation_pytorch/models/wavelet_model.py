import torch
import numpy as np
import torch.nn as nn
import segmentation_pytorch as smp


class WaveletModel(smp.UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_conv = nn.Conv2d(13, 3, kernel_size=1, stride=1)
        # inplanes = self.encoder.inplanes
        # layer0_modules = [
        #     ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
        #                         bias=False)),
        #     ('bn1', nn.BatchNorm2d(64)),
        #     ('relu1', nn.ReLU(inplace=True)),
        #     ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
        #                         bias=False)),
        #     ('bn2', nn.BatchNorm2d(64)),
        #     ('relu2', nn.ReLU(inplace=True)),
        #     ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
        #                         bias=False)),
        #     ('bn3', nn.BatchNorm2d(inplanes)),
        #     ('relu3', nn.ReLU(inplace=True)),
        # ]
        # self.encoder.layer0 = nn.Sequential(OrderedDict(layer0_modules))

    def forward(self, x):
        x = self.extra_conv(x)
        x = super().forward(x)
        return x
