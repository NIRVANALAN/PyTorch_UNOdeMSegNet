from typing import Any
import pdb
import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class PixelCELoss(nn.Module):
    __name__ = 'pixel_ce_loss'

    def __init__(self, normalize_size=False, num_classes=8):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction='none' if normalize_size else 'mean')
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
