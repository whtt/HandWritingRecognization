# /usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Intro  :
@Project: PR_HW
@File   : calc_mst.py
@Author : whtt
@Time   : 2020/10/8 19:00
"""


import os
import matplotlib.pyplot as plt
import numpy as np


# the root path
root_path = '.\\data'
# the path below root path, called label path
dir_path = os.listdir(root_path)

# each channel of image(R,G,B)
R_channel = 0
G_channel = 0
B_channel = 0

# search in the root path
for dir_ in dir_path:
    # path for each class(label)
    class_path = os.path.join(root_path, dir_)
    # path for images with above label
    file_path = os.listdir(class_path)
    # sum the image pixel by channel
    for file_ in file_path:
        file_name = os.path.join(class_path, file_)
        img = plt.imread(file_name) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

num = 100 * 420 * 420
# calculate the mean of the each channel
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

# set each channel as zero for the calculation of var
R_channel = 0
G_channel = 0
B_channel = 0

# search in the root path
for dir_ in dir_path:
    # path for each label
    class_path = os.path.join(root_path, dir_)
    # path for images with above label
    file_path = os.listdir(class_path)
    # sum the image pixel by channel
    for file_ in file_path:
        file_name = os.path.join(class_path, file_)
        img = plt.imread(file_name) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

# calculate the var by channel
R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)

print("mean[R G B]=[{} {} {}]".format(R_mean, G_mean, B_mean))
print("var[R G B]=[{} {} {}]".format(R_var, G_var, B_var))
