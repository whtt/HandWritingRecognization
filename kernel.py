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
from Fisher import VoteFisher, MultiFisher
from CNN import CNN
import torch
from torchvision import transforms as T
from PIL import Image


class Kernel:
    def __init__(self, log):
        log.info('loading methods ...')
        self.kernels = dict()
        self.naive_bayesian = NaiveBayes(log, pretrained=True, path='.\\Model\\naivebayes.pkl')
        self.vote_fisher = VoteFisher(log, pretrained=True, path='.\\Model\\ensemble_fisher.pkl')
        self.multi_fisher = MultiFisher(log, pretrained=True, path='.\\Model\\multi_class_fisher.pkl')
        sk_loader = torch.load('.\\Model\\sklearn_fisher.pkl')
        self.sklearn_fisher = sk_loader['fisher']
        self.skfisher_trans = sk_loader['transform']
        log.info('scikit-learn fisher model loaded')
        self.cnn = CNN()
        cnn_loader = torch.load('.\\Model\\cnn_best_.pkl')
        self.cnn.load_state_dict(cnn_loader['model_state'])
        self.cnn_trans = cnn_loader['transform']
        log.info('cnn model loaded')

    def set_kernel(self, im_path, method='NaiveBayesian'):
        image = Image.open(im_path)
        if method == 'NaiveBayesian':
            result = self.naive_bayesian.predict(image)
        elif method == 'VoteFisher':
            result = self.vote_fisher.predict(image)
        elif method == 'MultiFisher':
            result = self.multi_fisher.predict(image)
        elif method == 'SklearnFisher':
            result = self.sklearn_fisher.predict(self.skfisher_trans(image).flatten().reshape(1, 100))[0]
        elif method == 'CNN':
            logits, _ = self.cnn(1-self.cnn_trans(image)[0].reshape((1, 1, 28, 28)))
            result = torch.argmax(logits)
        else:
            raise ValueError('Invalid method')
        return result

