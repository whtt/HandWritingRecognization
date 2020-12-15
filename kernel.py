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


class Kernel:
    def __init__(self, log):
        self.kernels = dict()
        self.naive_bayesian = NaiveBayes(log)
        self.naive_bayesian.fit(data_loader('.\\data'))
        self.kernels['NaiveBayesian'] = self.naive_bayesian

    def set_kernel(self, method='NaiveBayesian'):
        return self.kernels[method]
