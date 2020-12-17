# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  :
@Project: PR_HW
@File   : CNN.py
@Author : whtt
@Time   : 2020/12/16 10:06
"""
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
from torch.utils import data
from torch.autograd import Variable
import tensorboardX
from torchvision.datasets import mnist


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = nn.Sequential(OrderedDict([
            ('ConvNet1', ConvNet(1, 16, 'ConvNet1', 3, 2, 1)),
            ('ConvNet2', ConvNet(16, 32, 'ConvNet2', 3, 2, 1)),
            ('ConvNet3', ConvNet(32, 64, 'ConvNet3', 2, 2, 0))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(2*2*64, 100)),
            ('FC2', nn.Linear(100, 10))
        ]))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def ConvNet(in_channels: int, out_channels: int, net_name: str, kernel_size=3, stride=1, padding=0):
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    norm = nn.BatchNorm2d(out_channels)
    active = nn.ReLU()
    net = nn.Sequential(OrderedDict([
        (net_name+'_conv', conv),
        (net_name+'_norm', norm),
        (net_name+'_relu', active)
    ]))
    return net


if __name__ =='__main__':
    mynet = CNN()
    print(mynet)
