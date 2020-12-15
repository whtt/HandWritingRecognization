# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  : some tools
@Project: PR_HW
@File   : utils.py
@Author : whtt
@Time   : 2020/10/5 13:18
"""
import os
from torchvision.datasets import ImageFolder
import numpy as np
# 使用logging打印日常调试信息，并保存到文件
import logging
from logging import handlers
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from Bayesian import NaiveBayes


def create_dir_path(path):
    """
    if the path given as parameter does not exist, create a directory(-ies) for the path
    :param path: path given to loop for
    :return:
    """
    if not os.path.isfile(path):
        if not os.path.exists(path):
            os.makedirs(path)
            return True
    return 0


def create_file_path(path):
    """
    create the path name for the handwriting number images, as each trace saved as '.txt'
    :param path:
    :return:
    """
    if os.path.isfile(path):
        return path
    num_class = path.split('/')[-1]
    file_num = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            file_num += 1
    file_path = os.path.join(path, num_class + '_{}.jpg'.format(file_num))
    return file_path


def data_loader(path):
    """
    load images
    :param path:
    :return:
    """
    # parameters of normalization: var, mean are calculated by file calc_mst.py
    # normalize = T.Normalize(mean=[0.918, 0.918, 0.918], std=[0.2, 0.2, 0.2])
    # transform of the images
    transform = T.Compose([
        T.RandomResizedCrop(size=28, scale=(0.9, 1.0)),
        # T.Grayscale(1),
        # T.RandomResizedCrop(28),  # random crop the image then resize to fixed size;
        # but if we crop the data, the image may lose their origin feature, so we need set the scale
        # T.RandomHorizontalFlip(),  # random flip the image;
        # but our data are fixed shape, flip has no effect for the data, so this is not need
        T.ToTensor(),  # change the image into tensor
        # normalize,
    ])
    # transform the image into tensors
    dataset = ImageFolder(path, transform=transform)
    # image = Image.open('.\\data\\0\\0_0.jpg')
    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    # plt.title("original image")
    # plt.subplot(2, 2, 2)
    # plt.imshow(T.RandomResizedCrop(size=14, scale=(0.8, 1.0))(image))
    # plt.title("random crop")
    # plt.subplot(2, 2, 3)
    # plt.imshow(T.RandomHorizontalFlip(p=1)(image))
    # plt.title("random flip")
    # plt.show()
    return dataset


def normalize(img):
    # plt.subplot(2, 1, 1)
    # plt.imshow(img)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            if img[i, j] > 0.8:
                img[i, j] = 1
            else:
                img[i, j] = 0
    # plt.subplot(2, 1, 2)
    # plt.imshow(img)
    # plt.show()
    return img


class Logger:
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(
            self,
            filename,
            level='info',
            when='D',
            back_count=3,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)  # 获取level信息
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置level级别
        sh = logging.StreamHandler()  # 往屏幕上输出日志
        sh.setFormatter(format_str)  # 设置屏幕上输出显示的格式
        th = handlers.TimedRotatingFileHandler(
            filename=filename,
            when=when,
            backupCount=back_count,
            encoding='utf-8'
        )  # 往文件里写入#when指定间隔时间自动生成文件的处理器
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 吧对象加到logger里
        self.logger.addHandler(th)


class Kernel:
    def __init__(self, log):
        self.kernels = dict()
        self.naive_bayesian = NaiveBayes(log)
        self.naive_bayesian.fit(data_loader('.\\data'))
        self.kernels['NaiveBayesian'] = self.naive_bayesian

    def set_kernel(self, method='NaiveBayesian'):
        return self.kernels[method]


if __name__ == '__main__':
    das = data_loader('.\\data')
    print(len(das))
