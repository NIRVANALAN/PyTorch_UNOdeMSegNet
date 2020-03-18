import torch
import pdb
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


class ExtremeLoss(nn.Module):
    __name__ = 'extreme_loss'

    def __init__(self):
        super().__init__()

    # def forward(self, *input):

    def forward(self, logpx, px_true):
        with torch.enable_grad():
            sum_class = 0.0
            for lpx, pxt in zip(logpx, px_true):
                px = torch.exp(lpx)
                true_idx = pxt > 0.5
                sum_class += - \
                    (torch.mean(lpx[true_idx]) - torch.mean(px[~true_idx]))
        return sum_class


class PixelCELoss(nn.Module):
    __name__ = 'pixel_ce_loss'

    def __init__(self, num_classes=8, weight=None, multi_stage=False):
        """

        :type weight: list
        """
        super().__init__()
        self.multi_stage = multi_stage
        if self.multi_stage:
            self.kldiv_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        if weight is not None:
            weight = torch.Tensor(weight)
            pass
        self.criterion = torch.nn.CrossEntropyLoss(
            reduction='mean', weight=weight)
        self.num_classes = num_classes

    @staticmethod
    def reshape_pred_label(pred, label):
        ps_pred = pred
        ps_label = label
        N, C, H, W = ps_pred.size()
        assert ps_label.size() == (N, H, W)
        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        ps_pred = ps_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()
        return ps_pred, ps_label

    def forward(self, pred, label, training=True):
        # Calculation
        if self.multi_stage and training:
            assert type(label) is list
            assert type(pred) is list
            assert len(label) == len(pred)
            assert all(
                [pred[i].shape == label[i].shape for i in range(len(pred) - 1)])
            stage_number = len(label)
            loss = 0.
            # for stage in range(stage_number):
            for stage in range(stage_number):
                loss += -(label[stage].to(pred[stage].dtype) * torch.log_softmax(pred[stage], dim=1)).mean(-1).mean(
                        -1).sum(
                    -1).mean(-1)
            loss /= stage_number
        # stage = -1
        # ps_pred, ps_label = self.reshape_pred_label(pred[stage], label[stage])
        # loss += self.criterion(ps_pred, ps_label)
        # for stage in range(0, stage_number - 1):
        # 	loss += self.kldiv_criterion(torch.log_softmax(pred[stage], dim=1), label[stage].to(pred[stage].dtype))
        # loss /= stage_number
        # print(loss)
        else:
            ps_pred, ps_label = self.reshape_pred_label(pred, label)
            loss = self.criterion(ps_pred, ps_label)
        return loss


# def __call__(self, pred, label):

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

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
        vert = torch.cat([F.conv2d(softmax[:, i].unsqueeze(
            1), vertical_sobel) for i in range(softmax.shape[0])], 1)
        horizontal = torch.cat(
            [F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
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


class PixelNLLLoss(nn.Module):
    __name__ = 'pixel_nll_loss'

    def __init__(self, normalize_size=False, num_classes=8, weight=None):
        """

        :type weight: list
        """
        super().__init__()
        if weight is not None:
            pass
        self.criterion = torch.nn.KLDivLoss(reduction='mean')
        self.num_classes = num_classes

    def forward(self, pred, label):
        # Calculation
        ps_pred = pred
        ps_label = label
        N, C, H, W = ps_pred.size()
        assert ps_label.size() == (N, C, H, W)  # BS * 8 * H * W
        ps_pred = F.log_softmax(ps_pred, 1)
        loss = self.criterion(ps_pred, ps_label)

        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        # pdb.set_trace()
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / \
            (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class MulticlassBCEDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassBCEDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        bcedice = BCEDiceLoss()
        totalLoss = 0
        for i in range(C):
            bcediceloss = bcedice(input[:, i], target[:, i])
            if weights is not None:
                bcediceloss *= weights[i]
            totalLoss += bcediceloss

        return totalLoss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
