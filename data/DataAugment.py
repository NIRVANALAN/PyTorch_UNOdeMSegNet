import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pywt

class WaveletDataAugmemt(Dataset):
    def __init__(self, dataset, wavelet, level, *args, **kwargs):
        self.dataset = dataset
        self.wavelet = wavelet
        self.level = level

        # self.target_channel = kwargs.get('target_channel', 0)


    def __len__(self):
        return self.dataset.__len__()


    def get_number_classes(self):
        return self.dataset.get_number_classes()


    def __getitem__(self, index):
        # get image
        img, label = self.dataset.__getitem__(index)
        img_size = img.shape[-2:]

        # compute wavelet decomposition
        coeffs = pywt.wavedec2(
            img[0],
            # img[self.target_channel],
            self.wavelet,
            level=self.level)
        coeff_list = [coeffs[0]]  # residual at first index
        for coeff in coeffs[1:]:
            coeff_list.extend(coeff)
        coeff_list = [cv2.resize(c, img_size, cv2.INTER_NEAREST) for c in coeff_list]
        coeff_list = torch.Tensor(coeff_list)

        # append to channels
        return torch.cat((img, coeff_list), axis=0), label