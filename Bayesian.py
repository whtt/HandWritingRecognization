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
import torch
from utils import Logger, data_loader, normalize


class NaiveBayes:
    def __init__(self, logger, pretrained=False, **kwargs):
        """
        NaiveBayes
        i. p(c|x) = p(c)/p(x) * \\prod_{i=1}^{d} p(x_i|c)
            for c: classes; x: features; x_i: feature_i
        ii. p(c) = \abs(D_c) / \abs(D)
            for D_c: nums of classes c; D: num of all
        iii. p(x_i|c) = \abs(D_c, x_i) / \abs(D_c)
        iv. p(x): normalized param, same to all, set to 1
        v. Laplacian correction: ii.&iii. can be corrected as:
        \frac{*+1}{*+N_i or N}
            for N: cnt(label \\in D); N_i: cnt(x_i \\in D_c)
        :param logger:
        :param pretrained:
        :param kwargs:
        """
        # log for printing information
        self.log = logger
        self.transform = None
        # length for feature dims, length=-1 means untrained
        self.length = -1
        # label probability p(c)
        self.label_prob = dict()
        # feature probability p(x|c)
        self.feature_prob = dict()
        # pretrained model can be loaded here
        if pretrained:
            # show that load process begins
            self.log.info('data pretrained, ')
            self.load_model(kwargs['path'])

    def fit(self, data_load):
        dataset, self.transform = data_load
        # show that train porcess begins
        self.log.info('====== training start ======')
        # the length of feature dims
        self.length = len(dataset[0][0].flatten())
        # labels nums, the length of dataset
        labels_num = len(dataset.targets)
        # label kinds
        classes_num = set(dataset.classes)
        # create and set the label prob p(c) = cnt(D_c)/cnt(D), actually unused
        for item in classes_num:
            self.label_prob[item] = dataset.targets.count(int(item))/labels_num
        for num, (img, label) in enumerate(dataset):
            # if p(x|c) is not exist, create it
            if str(label) not in self.feature_prob:
                self.feature_prob[str(label)] = []
            # transform the image(10x10) into 1-dim array
            feature_vector = img.flatten()
            self.feature_prob[str(label)].append(feature_vector)
        self.log.info('====== training over ======')

    def test(self, data_load):
        dataset, _ = data_load
        if self.length == -1:
            raise ValueError("Please train the model")
        acc = 0
        cnt = 0
        self.log.info('====== testing start ======')
        for num, (img, label) in enumerate(dataset):
            predicted = self.predict(img, is_tensor=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label, predicted, bool(label == predicted)))
            if label == predicted:
                acc += 1
            cnt += 1
        acc = acc * 100.0 / cnt
        self.log.info('the top1 accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def predict(self, image, is_tensor=False, top5=False):
        if self.length == -1:
            raise ValueError("Please train the model")
        # set a prob list: p(c_i|x)
        result = dict()
        for label_ in range(10):
            label = str(label_)
            # p(c_i)=0.1; c_i: label
            # labels = self.label_prob[label]
            # p(x|c_i); x: feature
            features = self.feature_prob[label]
            # p(x_j|c_i); x_j: feature[index]
            # p = p(x_1|c_i) * ... * p(x_d|c_i)
            # p(x_j|c_i) = |D_c, x_i| / |D_c|
            # p *= p_ 太小向下溢出; instead: log（p）
            # using Laplacian correction
            features = torch.tensor(np.array([np.array(feature) for feature in features]))
            if is_tensor:
                vector = image.flatten()
            else:
                vector = self.transform(image).flatten()
            # compare all training data with testing one by pixel, then sum by pixel
            counts = torch.eq(features, vector).sum(dim=0)
            # \\prod p(x_j|c)
            p = torch.log((counts+1)*(1/(features.shape[0]+2))).sum()
            # p(c_i|x) = p(c) * \\prod_{i=1}^{d} p(x_i|c)
            result[label] = p.item()
        self.log.debug('result of the image is: {}'.format(result))
        if top5:
            predicted = np.argsort(list(result.values()))
            return predicted[5:]
        else:
            predicted = np.argmax(list(result.values()))
            return int(predicted)

    def top5rate(self, data_load):
        dataset, _ = data_load
        if self.length == -1:
            raise ValueError("Please train the model")
        acc = 0
        cnt = 0
        self.log.info('====== testing start ======')
        for num, (img, label) in enumerate(dataset):
            predicted = self.predict(img, is_tensor=True, top5=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label, predicted, bool(label in predicted)))
            if label in predicted:
                acc += 1
            cnt += 1
        acc = acc * 100.0 / cnt
        self.log.info('the top5 accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def save_model(self, path):
        torch.save(
            {
                'length': self.length,
                'feature_prob': self.feature_prob,
                'transform': self.transform
            },
            path
        )
        self.log.info('model saved in path:{}'.format(os.path.abspath(path)))

    def load_model(self, path):
        params = torch.load(path)
        self.length = params['length']
        self.feature_prob = params['feature_prob']
        self.transform = params['transform']
        self.log.info('NaiveBayesian model loaded')


if __name__ == '__main__':
    log = Logger('logs/bayesian/Bayesian.log', level='debug')
    my_net = NaiveBayes(log.logger)
    my_net.fit(data_loader('.\\data'))
    my_net.test(data_loader('.\\test'))
    my_net.top5rate(data_loader('.\\test'))
    my_net.save_model('.\\Model\\naivebayes.pkl')
