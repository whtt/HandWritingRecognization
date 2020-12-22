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
import torch
# 使用logging打印日常调试信息，并保存到文件
import logging
from logging import handlers
from torchvision import transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np


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


def data_loader(file_path, size=(10, 10), threshold=0.68, equal: bool = False, deep: bool = False):
    """
    load images
    :param file_path:
    :param size:
    :param threshold:
    :param equal:
    :param deep:
    :return:
    """
    if deep:
        if not equal:
            raise ValueError('segment should be equal')
        if type(size) != int:
            raise TypeError('size should be int, got {} instead'.format(type(size)))
        transform = T.Compose([
            ImSegment(equal=equal, deep=deep),
            T.Resize(size),
            T.ToTensor(),
        ])
    else:
        if equal and type(size) != int:
            raise TypeError('the equal seg image size suppose to be int, got type {} instead'.format(type(size)))
        if (not equal) and type(size) == int:
            raise TypeError('the segment image shape is supposed unequal, but got equal size {}({}) instead'.format(
                type(size), size
            ))
        # parameters of normalization: var, mean are calculated by file calc_mst.py
        # normalize = T.Normalize(mean=[0.918, 0.918, 0.918], std=[0.2, 0.2, 0.2])
        # transform of the images
        transform = T.Compose([
            ImSegment(equal=equal),
            T.Resize(size),
            T.ToTensor(),
            Normal(threshold=threshold),
        ])
    # transform the image into tensors
    dataset = ImageFolder(file_path, transform=transform)
    # image = Image.open('.\\data\\0\\0_0.jpg')
    # plt.subplot(2, 2, 1)
    # plt.imshow(image)
    # plt.title("original image")
    # plt.subplot(2, 2, 2)
    # plt.imshow(ImSegment(equal=equal)(image))
    # plt.title("random crop")
    # plt.subplot(2, 2, 3)
    # plt.imshow(T.RandomHorizontalFlip(p=1)(image))
    # plt.title("random flip")
    # plt.show()
    return dataset, transform


class ImSegment:
    def __init__(self, equal: bool = True, deep: bool = False):
        if (not equal) and deep:
            raise ValueError('the param deep is designed to deep learning methods, require equal = True')
        self.equal = equal
        self.deep = deep

    def __call__(self, img):
        # flip the img by color(0<-->255) ==> then we can use methods to crop the image
        img_flip = ImageOps.invert(img)
        # sum up the pixels by row/col, to capture the attention area
        row_sum = np.sum(img_flip, axis=1)[:, 0]
        col_sum = np.sum(img_flip, axis=0)[:, 0]
        # get the index of the attention area position
        row_min = (row_sum != 0).argmax()
        row_max = (row_sum != 0)[::-1].argmax()
        col_min = (col_sum != 0).argmax()
        col_max = (col_sum != 0)[::-1].argmax()
        # crop the attention area
        im_attention = ImageOps.crop(img, (col_min, row_min, col_max, row_max))
        # get the shape of the new image
        row, col = im_attention.size
        # create the new image
        # if equal, make a square image
        # if not, make a rectangle
        if self.equal:
            if self.deep:
                im_new = Image.new(img.mode, (420, 420), color='white')
                im_new.paste(im_attention, (int(210-row/2), int(210-col/2)))
            else:
                # create a new image -- white
                im_new = Image.new(img.mode, (max(row, col), max(row, col)), color='white')
                # fit the attention area into the new image, on center
                if row > col:
                    im_new.paste(im_attention, (0, int((row - col)/2)))
                else:
                    im_new.paste(im_attention, (int((col - row)/2), 0))
        else:
            im_new = im_attention
        return im_new


class Normal:
    def __init__(self, threshold=0.68):
        self.threshold = threshold

    def __call__(self, im_tensor_):
        im_tensor_[im_tensor_ > self.threshold] = 1
        im_tensor_[im_tensor_ < self.threshold] = 0
        return im_tensor_[0]


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


if __name__ == '__main__':
    for i in range(10):
        for j in range(20):
            path = '.\\data\\{}\\{}_{}.jpg'.format(i, i, j)
            image = Image.open(path)
            # plt.subplot(2, 2, 1)
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.title("original image")

            transform = T.Compose([
                ImSegment(equal=True, deep=True),
                T.Resize(28),
                T.ToTensor(),
            ])

            # plt.subplot(1, 2, 2)
            plt.imshow((1-transform(image)[0]), plt.gray())
            plt.title("transformed image")

            # im_seg = ImSegment(equal=False)(image)
            # plt.subplot(2, 2, 2)
            # plt.imshow(im_seg)
            # plt.title("Segment Image")
            #
            # im_resize = T.Resize((10, 10))(im_seg)
            # plt.subplot(2, 2, 3)
            # plt.imshow(im_resize)
            # plt.title("resize the image")
            #
            # im_tensor = T.ToTensor()(im_resize)
            # im_norm = Normal(threshold=0.68)(im_tensor)
            #
            # plt.subplot(2, 2, 4)
            # plt.imshow(im_norm, plt.gray())
            # plt.title("final normed ")

            plt.show()
