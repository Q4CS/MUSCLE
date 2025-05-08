# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


class AutomaticWeightedLossCipolla(nn.Module):
    """
    [1] R. Cipolla, Y. Gal, and A. Kendall, “Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Salt Lake City, UT, USA: IEEE, Jun. 2018, pp. 7482–7491. doi: 10.1109/CVPR.2018.00781.

    automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLossCipolla(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLossCipolla, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        params_list = []
        for i, loss in enumerate(x):
            temp_weight = torch.clamp(self.params[i], 1e-5, 1e+5)
            loss_sum += 0.5 / (temp_weight ** 2) * loss + torch.log(temp_weight)
            params_list.append(temp_weight.item())

        return loss_sum, params_list


class AutomaticWeightedLossLiebel(nn.Module):
    """
    [1] L. Liebel and M. Körner, “Auxiliary Tasks in Multi-task Learning,” May 17, 2018, arXiv: arXiv:1805.06334. Accessed: Sep. 06, 2024. [Online]. Available: http://arxiv.org/abs/1805.06334

    automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLossLiebel(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLossLiebel, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        params_list = []
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            params_list.append(self.params[i].item())

        return loss_sum, params_list


class AutoMultipleWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutoMultipleWeightedLoss, self).__init__()
        params = torch.zeros(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        params_list = []
        for i, loss in enumerate(x):
            loss_sum += torch.sigmoid(self.params[i]) * loss
            params_list.append(torch.sigmoid(self.params[i]).item())
        return loss_sum, params_list


class AutoOneWeightedLoss(nn.Module):
    def __init__(self, ):
        super(AutoOneWeightedLoss, self).__init__()
        params = torch.tensor(0.0, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x1, x2):
        lambda_ = torch.sigmoid(self.params)
        loss_sum = lambda_ * x1 + (1 - lambda_) * x2
        return loss_sum, [lambda_.item()]


class FixedWeightLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(FixedWeightLoss, self).__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.beta = torch.tensor(beta, requires_grad=False)

    def forward(self, x1, x2):
        loss_sum = self.alpha * x1 + self.beta * x2
        return loss_sum, [self.alpha.item(), self.beta.item()]
