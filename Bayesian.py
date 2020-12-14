# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  : NaiveBayes for handwriting recognition
@Project: PR_HW
@File   : Bayesian.py
@Author : whtt
@Time   : 2020/10/6 18:23
"""


import numpy as np
import os
import time
from utils import Logger, data_loader


class Bayes:
    def __init__(self, logger, pretrained=False, **kwargs):
        # log for printing information
        self.log = logger
        # length for feature dims, length=-1 means untrained
        self.length = -1
        # label probability
        self.label_prob = dict()
        #
        self.feature_prob = dict()
        if pretrained:
            self.log.info('data pretrained, ')
            self.length = 1

    def fit(self, dataset):
        self.log.info('train started')
        self.length = len(dataset[0][0].squeeze)
        labels_num = len(dataset.targets)
        classes_num = set(dataset.classes)
        for item in classes_num:
            self.label_prob[item] = dataset.targets.count(item)/labels_num
        for num, img, label in enumerate(dataset):
            if label not in self.feature_prob:
                self.feature_prob[label] = []

    def predict(self):
        return 1


if __name__ == '__main__':
    log = Logger('logs/Bayesian.log', level='debug')
    my_net = Bayes(log.logger)
    my_net.fit(data_loader('.\\data'))
