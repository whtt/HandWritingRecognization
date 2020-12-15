# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  :
@Project: PR_HW
@File   : kernel.py
@Author : whtt
@Time   : 2020/12/15 13:51
"""

from Bayesian import NaiveBayes
from utils import data_loader
from torchvision import transforms as T
from PIL import Image


class Kernel:
    def __init__(self, log):
        self.kernels = dict()
        self.naive_bayesian = NaiveBayes(log)
        self.naive_bayesian.fit(data_loader('.\\data'))

    def set_kernel(self, im_path, method='NaiveBayesian'):
        image = Image.open(im_path)
        image = T.RandomResizedCrop(28, scale=(0.8, 1.0))(image)
        image = T.ToTensor()(image)
        if method == 'NaiveBayesian':
            result = self.naive_bayesian.predict(image)
            return result
