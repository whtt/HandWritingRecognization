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
import os
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from utils import Logger, data_loader, normalize


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature = nn.Sequential(OrderedDict([
            ('ConvNet1', ConvNet(1, 16, 'ConvNet1', 3, 2, 1)),
            ('ConvNet2', ConvNet(16, 32, 'ConvNet2', 3, 2, 1)),
            ('ConvNet3', ConvNet(32, 64, 'ConvNet3', 2, 2, 0))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(576, 100)),
            ('FC2', nn.Linear(100, 10)),
            ('SoftMax', nn.Softmax())
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


if __name__ == '__main__':
    lr = 0.001
    weight_decay = 1e-4
    epochs = 200
    batch_size = 128
    log = Logger('.\\logs\\CNN.log', level='debug')

    # Mnist digits dataset
    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True
    else:
        DOWNLOAD_MNIST = False

    # download the mnist dataset
    train_data = MNIST(root='.\\mnist\\train', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    test_data = MNIST(root='.\\mnist\\train', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)

    # plot one example
    print(train_data.data.size())  # (60000, 28, 28)
    print(train_data.targets.size())  # (60000)
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('%i' % train_data.targets[0])
    plt.show()

    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
    test_y = test_data.targets[:2000]

    mynet = CNN()
    # print(mynet)
    log.logger.info(mynet)

    optimizer = torch.optim.Adam(mynet.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for step, (train_x, train_y) in enumerate(train_loader):
            mynet.train()
            logits = mynet(train_x)
            loss = criterion(logits, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                mynet.eval()
                test_output = mynet(test_x)
                test_loss = criterion(test_output, test_y)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                acc = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                log.logger.info(
                    'Epoch:[{:3}|{:3}] | step:{:3} | train_loss:{:.4f} | test_loss:{:.4f} | test_acc:{:.2f}%'.format(
                        epoch, epochs, step, loss, test_loss, acc*100
                    )
                )

    mynet.eval()
    train_x_all = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)/255
    train_y_all = train_data.targets
    test_x_all = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255
    test_y_all = test_data.targets

    train_logits = mynet(train_x_all)
    test_logits = mynet(test_x_all)
    train_loss_last = criterion(train_logits, train_y_all)
    test_loss_last = criterion(test_logits, test_y_all)
    pred_train_all = torch.max(train_logits, 1)[1].data.numpy()
    pred_test_all = torch.max(test_logits, 1)[1].data.numpy()
    acc_train = float((pred_train_all == train_y_all.data.numpy()).astype(int).sum()) / float(train_y_all.size(0))
    acc_test = float((pred_test_all == test_y_all.data.numpy()).astype(int).sum()) / float(test_y_all.size(0))
    log.logger.info(
        'train loss: {:.4f} | test loss: {:.4f} | train acc: {:.2f}% | test acc: {:.2f}%'.format(
            train_loss_last, test_loss_last, acc_train*100, acc_test*100
        )
    )

    my_dataset = data_loader('.\\data')
    cnt = 0
    acc_ = 0
    for img, label in my_dataset:
        result = mynet(normalize(1-img[0]).reshape(1, 1, 28, 28))
        predicted = torch.max(result, 1)[1].data
        if predicted == label:
            acc_ += 1
        cnt += 1
    acc_ = acc_ * 100.0 / cnt
    log.logger.info('the accuracy of our own handwriting data is {}%'.format(acc_))
