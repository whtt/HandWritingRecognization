# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  :
@Project: PR_HW
@File   : Fisher.py
@Author : whtt
@Time   : 2020/12/15 17:25
"""


import torch
import numpy as np
import os
from utils import data_loader, Logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class BiFisher:
    def __init__(self, dataset, labels, logger):
        # printer
        self.log = logger
        # data, label initial
        self.data = dataset
        self.labels = labels
        self.length = len(dataset[0][0].flatten())
        # self.data = torch.zeros((1, 784)).float()
        # self.labels = torch.zeros((1, 1)).int()
        # mean of each feature of each class
        self.mean_all = torch.zeros((1, self.length)).float()
        self.mean = torch.zeros((2, 1, self.length)).float()
        # std of each feature of each class
        self.conv = torch.zeros((2, self.length, self.length)).float()
        # linear discriminator parameter
        self.weight = torch.zeros((self.length, 1)).float()
        # within-class scatter matrix
        self.s_w = torch.zeros((self.length, self.length)).float()
        # between-class scatter matrix
        self.s_b = torch.zeros((self.length, self.length)).float()
        # center of each data block
        self.center = torch.zeros((2, ))

    def fit(self, label_1: int, label_0: int):
        self.log.info("====== training start ======")
        # for img, label in dataset:
        #     self.data = torch.cat((self.data, normalize(img[0]).flatten()))
        #     self.labels = torch.cat((self.labels, torch.tensor([[int(label)]])))
        # delete the first data(zero element)
        # self.data = self.data[1:]
        # self.labels = self.labels[1:]
        # calculate the mean of each class
        # choose the data belong to label given(true one)
        data_true = self.data[self.labels.eq(label_1)]
        data_false = self.data[self.labels.eq(label_0)]
        # calculate the mean of label true
        self.mean_all = torch.reshape(torch.mean(self.data, dim=0), shape=(1, self.length))
        self.mean[0] = torch.mean(data_false, keepdim=True, dim=0)
        self.mean[1] = torch.mean(data_true, keepdim=True, dim=0)
        # (x-u_i).T
        delta_data_false = data_false - self.mean[0]
        delta_data_true = data_true - self.mean[1]
        # s_w_i = \sum (x-u_i)*(x-u_i).T
        for line in range(delta_data_false.shape[0]):
            self.conv[0] += torch.mm(delta_data_false[line].T, delta_data_false[line])
        for line in range(delta_data_true.shape[0]):
            self.conv[1] += torch.mm(delta_data_true[line].T, delta_data_true[line])
        num = self.data[self.labels.eq(label_0)].shape[0]
        self.s_b += num * torch.mm((self.mean[0] - self.mean_all).T, (self.mean[0] - self.mean_all))
        num = self.data[self.labels.eq(label_1)].shape[0]
        self.s_b += num * torch.mm((self.mean[1] - self.mean_all).T, (self.mean[1] - self.mean_all))
        # calculate the s_w = \sum (s_w_i)
        self.s_w = torch.sum(self.conv, dim=0)
        # use singular value decomposition(SVD) instead inv: s_w = U * S * V
        U, S, V = torch.svd(self.s_w)
        # s_w_inv = V.T * S_inv * V.T
        s_w_inv = torch.mm(torch.mm(V.T, torch.inverse(torch.diag(S))), U.T) * self.s_b
        # s_w_inv = torch.mm(torch.mm(V.T, torch.inverse(torch.diag(S))), U.T)
        # w = s_w_inv * (u1-u0): (true-false)
        self.weight = torch.mm(s_w_inv, self.mean[1].T - self.mean[0].T)
        # calculate the center of each block: x_bar = w.T * u
        self.center[0] = torch.mm(self.weight.T, self.mean[0].T)
        self.center[1] = torch.mm(self.weight.T, self.mean[1].T)
        self.log.info('====== training over ======')

    def predict(self, image, label_1: int, label_0: int):
        # calculate the position of the data
        pos = torch.mm(self.weight.T, image.flatten().reshape(self.length, 1))
        result = torch.tensor([label_1, label_0])
        return result[np.argmin([torch.abs(pos - self.center[1]), torch.abs(pos - self.center[0])])]

    def test(self, data_load, label_1: int, label_0: int):
        dataset, _ = data_load
        cnt = 0
        acc = 0
        self.log.info('====== testing start ======')
        for img, label in dataset:
            if int(label) in [label_0, label_1]:
                predicted = self.predict(img, label_1, label_0)
                self.log.info('true:{}|pred:{}|result:{}'.format(label, predicted, bool(label == predicted)))
                cnt += 1
                if predicted == label:
                    acc += 1
        acc = acc * 100.0 / cnt
        self.log.info('accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')
        return acc/100.0

    def param_info(self):
        return self.weight, self.center


class VoteFisher:
    def __init__(self, logger, pretrained=False, **kwargs):
        self.log = logger
        self.length = -1
        # # design ten classifer, calculate the final sum, devote a best one
        # self.votes = torch.zeros((10, ))
        self.classfiers = list()
        self.transform = None
        if pretrained:
            # show that load process begins
            self.log.info('data pretrained, ')
            self.load_model(kwargs['path'])

    def fit(self, data_load):
        dataset, self.transform = data_load
        self.length = len(dataset[0][0].flatten())
        data = torch.zeros((1, 1, self.length)).float()
        labels = torch.zeros((1, )).int()
        self.log.info('fitting data')
        for img, label in dataset:
            data = torch.cat((data, img.flatten().reshape(1, 1, self.length)))
            labels = torch.cat((labels, torch.tensor([int(label)])))
        # delete the first data(zero element)
        data = data[1:]
        labels = labels[1:]
        for i in range(10):
            for j in range(10):
                self.log.info('train the [{}|{}] model: label[{}vs{}]'.format(i*10+j+1, 100, i, j))
                my_fisher = BiFisher(data, labels, self.log)
                my_fisher.fit(label_1=i, label_0=j)
                v = my_fisher.test((dataset, self.transform), label_1=i, label_0=j)
                # my_fisher.test(data_loader('.\\test'), label_1=i, label_0=j)
                w, c = my_fisher.param_info()
                self.classfiers.append([w, c, v])
                self.log.info('the model [{}|{}] train over'.format(i*10+j+1, 100))

    def predict(self, image, is_tensor=False, top5=False):
        voter = torch.zeros((10,))
        cla_num = 0
        for i in range(10):
            for j in range(10):
                w, c, v = self.classfiers[cla_num]
                if is_tensor:
                    pos = torch.mm(w.T, image.flatten().reshape(self.length, 1))
                else:
                    pos = torch.mm(w.T, self.transform(image).flatten().reshape(self.length, 1))
                voted = np.argmin([torch.abs(pos - c[1]), torch.abs(pos - c[0])])
                if voted == 0:
                    voter[i] += v
                else:
                    voter[j] += v
                cla_num += 1
        if top5:
            predicted = torch.argsort(voter)
            return predicted[5:]
        else:
            predicted = torch.argmax(voter)
            return predicted

    def test(self, data_load):
        dataset, _ = data_load
        cnt = 0
        acc = 0
        self.log.info('====== testing start ======')
        for img, label in dataset:
            predicted = self.predict(img, is_tensor=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label, predicted, bool(label == predicted)))
            if label == predicted:
                acc += 1
            cnt += 1
        acc = acc * 100 / cnt
        self.log.info('the top1 accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def top5rate(self, data_load):
        dataset, _ = data_load
        cnt = 0
        acc = 0
        self.log.info('====== testing start ======')
        for img, label in dataset:
            predicted = self.predict(img, is_tensor=True, top5=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label, predicted, bool(label in predicted)))
            if label in predicted:
                acc += 1
            cnt += 1
        acc = acc * 100 / cnt
        self.log.info('the top5 accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def load_model(self, path):
        model = torch.load(path)
        self.classfiers = model['classifiers']
        self.transform = model['transform']
        self.length = model['length']
        self.log.info('VoteFisher model loaded')

    def save_model(self, path):
        torch.save(
            {
                'classifiers': self.classfiers,
                'transform': self.transform,
                'length': self.length
            },
            path
        )
        self.log.info('model saved in path:{}'.format(os.path.abspath(path)))


class MultiFisher:
    def __init__(self, logger, pretrained=False, **kwargs):
        self.log = logger
        self.length = -1
        self.transform = None
        # # std of each feature of each class
        # self.conv = torch.zeros((2, self.length, self.length)).float()
        # # linear discriminator parameter
        self.weight = None
        # # within-class scatter matrix
        # self.s_w = torch.zeros((self.length, self.length)).float()
        # # between-class scatter matrix
        # self.s_b = torch.zeros((self.length, self.length)).float()
        # # center of each data block
        self.center = torch.zeros((10, 10))
        if pretrained:
            # show that load process begins
            self.log.info('data pretrained, ')
            self.load_model(kwargs['path'])

    def fit(self, data_load):
        dataset, self.transform = data_load
        self.length = len(dataset[0][0].flatten())
        data_ = torch.zeros((1, 1, self.length)).float()
        labels_ = torch.zeros((1,)).int()
        mean = torch.zeros((10, self.length)).float()
        conv = torch.zeros((10, self.length, self.length)).float()
        s_b = torch.zeros((self.length, self.length)).float()
        self.log.info("====== training start ======")
        for img_, label_ in dataset:
            data_ = torch.cat((data_, img_.flatten().reshape(1, 1, self.length)))
            labels_ = torch.cat((labels_, torch.tensor([int(label_)])))
        # delete the first data(zero element)
        data_ = data_[1:]
        labels_ = labels_[1:]
        # calculate the mean of each class
        # calculate the mean of label true
        mean_all = torch.reshape(torch.mean(data_, dim=0), shape=(1, self.length))
        for i in range(10):
            mean[i] = torch.mean(data_[labels_.eq(i)], keepdim=True, dim=0)
            # (x-u_i).T
            delta_data = data_[labels_.eq(i)] - mean[i]
            # s_w_i = \sum (x-u_i)*(x-u_i).T
            for line in range(delta_data.shape[0]):
                conv[i] += torch.mm(delta_data[line].T, delta_data[line])
            num = data_[labels_.eq(i)].shape[0]
            s_b += num * torch.mm((mean[i] - mean_all).T, (mean[i] - mean_all))
        # calculate the s_w = \sum (s_w_i)
        s_w = torch.sum(conv, dim=0)
        # use singular value decomposition(SVD) instead inv: s_w = U * S * V
        U, S, V = torch.svd(s_w)
        # s_w_inv = V.T * S_inv * V.T
        s_w_inv = torch.mm(torch.mm(V.T, torch.inverse(torch.diag(S))), U.T) * s_b
        # obtain the main eigen of s_W^-1 * s_b
        eigenvalues, eigenvectors = torch.eig(s_w_inv, eigenvectors=True)
        eigen_abs = torch.sqrt(torch.sum(torch.mul(eigenvalues, eigenvalues), dim=1))
        eigen_index = torch.argsort(eigen_abs)[-int(self.length*0.5):]
        # w = s_w_inv * (u1-u0): (true-false)
        self.weight = eigenvectors[eigen_index]
        # calculate the center of each block: x_bar = w.T * u
        self.center = torch.mm(self.weight, mean.T)
        self.log.info('====== training over ======')

    def predict(self, image, is_tensor=False, top5=False):
        # calculate the position of the data
        if is_tensor:
            pos = torch.mm(self.weight, image.flatten().reshape(self.length, 1))
        else:
            pos = torch.mm(self.weight, self.transform(image).flatten().reshape(self.length, 1))
        delta_pos = self.center - pos
        # calculate the distance
        dis_pos = torch.sum(torch.mul(delta_pos, delta_pos), dim=0)
        if top5:
            result = torch.argsort(dis_pos)
            return result[:5]
        else:
            result = torch.argmin(dis_pos)
            return result

    def test(self, data_load):
        dataset, _ = data_load
        cnt = 0
        acc = 0
        self.log.info('====== testing start ======')
        for img_, label_ in dataset:
            predicted = self.predict(img_, is_tensor=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label_, predicted, bool(label_ == predicted)))
            cnt += 1
            if predicted == label_:
                acc += 1
        acc = acc * 100.0 / cnt
        self.log.info('accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def top5rate(self, data_load):
        dataset, _ = data_load
        if self.length == -1:
            raise ValueError("Please train the model")
        acc = 0
        cnt = 0
        self.log.info('====== testing start ======')
        for num, (img_, label_) in enumerate(dataset):
            predicted = self.predict(img_, is_tensor=True, top5=True)
            self.log.info('true:{}|pred:{}|result:{}'.format(label_, predicted, bool(label_ in predicted)))
            if label_ in predicted:
                acc += 1
            cnt += 1
        acc = acc * 100.0 / cnt
        self.log.info('the top5 accuracy of the dataset is: {}%'.format(acc))
        self.log.info('====== testing over ======')

    def load_model(self, path):
        model = torch.load(path)
        self.center = model['center']
        self.transform = model['transform']
        self.length = model['length']
        self.weight = model['weight']
        self.log.info('MultiFisher model loaded')

    def save_model(self, path):
        torch.save(
            {
                'center': self.center,
                'transform': self.transform,
                'length': self.length,
                'weight': self.weight
            },
            path
        )
        self.log.info('model saved in path:{}'.format(os.path.abspath(path)))


if __name__ == '__main__':
    log = Logger('logs/fisher/Fisher.log', level='debug')
    mult_net = MultiFisher(log.logger)
    mult_net.fit(data_loader('.\\data'))
    mult_net.test(data_loader('.\\test'))
    mult_net.top5rate(data_loader('.\\test'))
    mult_net.save_model('.\\Model\\multi_class_fisher.pkl')

    my_net = VoteFisher(log.logger)
    my_net.fit(data_loader('.\\data', size=(7, 7), threshold=0.65))
    my_net.test(data_loader('.\\test', size=(7, 7), threshold=0.65))
    my_net.top5rate(data_loader('.\\test', size=(7, 7), threshold=0.65))
    my_net.save_model('.\\Model\\ensemble_fisher.pkl')

    train_data, transform_sk = data_loader('.\\data')
    data = np.zeros((1, 100))
    labels = np.zeros(1, dtype=np.int)
    for img, label in train_data:
        data = np.concatenate((data, img.flatten().reshape(1, 100).numpy()), axis=0)
        labels = np.concatenate((labels, np.array([int(label)], dtype=np.int)), axis=0)
    test_data, _ = data_loader('.\\test')
    data_test = np.zeros((1, 100))
    labels_test = np.zeros(1, dtype=np.int)
    for img, label in test_data:
        data_test = np.concatenate((data_test, img.flatten().reshape(1, 100).numpy()), axis=0)
        labels_test = np.concatenate((labels_test, np.array([int(label)], dtype=np.int)), axis=0)
    data = data[1:, :]
    labels = labels[1:]
    data_test = data_test[1:, :]
    labels_test = labels_test[1:]
    fisher = LinearDiscriminantAnalysis()
    fisher.fit(data, labels)
    torch.save(
        {
            'fisher': fisher,
            'transform': transform_sk,
        },
        '.\\Model\\sklearn_fisher.pkl'
    )
    predict_train = fisher.predict(data)
    print(predict_train)
    predict_test = fisher.predict(data_test)
    print(np.sum(predict_test == labels_test)/len(labels_test))


