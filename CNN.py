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
from tensorboardX import SummaryWriter
import os
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from utils import Logger, data_loader


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
        logits = self.classifier(x)
        return logits, x


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
    # lr = 0.001
    # weight_decay = 1e-4
    # epochs = 100
    # batch_size = 128
    log = Logger('.\\logs\\cnn\\CNN_.log', level='debug')
    # writer = SummaryWriter(comment='CNN')
    model_path = '.\\Model\\cnn_.pkl'
    best_model_path = '.\\Model\\cnn_best_.pkl'
    my_dataset, transform = data_loader('.\\data', size=28, equal=True, deep=True)

    # Mnist digits dataset
    if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        # not mnist dir or mnist is empyt dir
        DOWNLOAD_MNIST = True
    else:
        DOWNLOAD_MNIST = False

    # download the mnist dataset
    # train_data = MNIST(root='.\\mnist\\train', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    test_data = MNIST(root='.\\mnist\\train', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
    #
    # # plot one example
    # print(train_data.data.size())  # (60000, 28, 28)
    # print(train_data.targets.size())  # (60000)
    # plt.imshow(train_data.data[0].numpy(), cmap='gray')
    # plt.title('%i' % train_data.targets[0])
    # plt.show()
    #
    # train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255
    test_y = test_data.targets[:2000]
    #
    # plot one example
    # for i in range(2000):
    #     plt.imshow(test_x[i].reshape((28, 28)).numpy(), cmap='gray')
    #     plt.title('%i' % test_y[i])
    #     plt.show()
    #
    # dumpy_input = torch.rand(13, 1, 28, 28)
    #
    # mynet = CNN()
    # # print(mynet)
    # log.logger.info(mynet)
    # writer.add_graph(mynet, ((dumpy_input, )))
    #
    # optimizer = torch.optim.Adam(mynet.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    #
    # total_step = 0
    # acc_best = 0
    # for epoch in range(epochs):
    #     for step, (train_x, train_y) in enumerate(train_loader):
    #         mynet.train()
    #         logits, _ = mynet(train_x)
    #         loss = criterion(logits, train_y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         writer.add_scalar('train/loss', loss, total_step)
    #         # writer.add_image(
    #         #     'bottleneck_step{}_label{}'.format(total_step, train_y[0]), bottle_neck[0].reshape((1, 24, 24)),
    #         #     total_step
    #         # )
    #         total_step += 1
    #
    #         if step % 50 == 0:
    #             mynet.eval()
    #             test_output, bottle_neck = mynet(test_x)
    #             test_loss = criterion(test_output, test_y)
    #             pred_y = torch.max(test_output, 1)[1].data.numpy()
    #             acc = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    #             log.logger.info(
    #                 'Epoch:[{:3}|{:3}] | step:{:3} | train_loss:{:.4f} | test_loss:{:.4f} | test_acc:{:.2f}%'.format(
    #                     epoch, epochs, step, loss, test_loss, acc*100
    #                 )
    #             )
    #             writer.add_scalar('test/loss', test_loss, total_step)
    #             for i in range(16):
    #                 writer.add_image('origin_image{}'.format(i), test_x[100+i].reshape(1, 28, 28), total_step)
    #                 writer.add_image('BottleNeck_im{}'.format(i), bottle_neck[100+i].reshape(1, 24, 24), total_step)
    #             torch.save(
    #                 {
    #                     'model_state': mynet.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     'loss': loss,
    #                     'epoch': epoch,
    #                     'lr': lr,
    #                     'wc': weight_decay,
    #                     'transform': transform,
    #                  },
    #                 model_path
    #             )
    #             if acc > acc_best:
    #                 acc_best = acc
    #                 torch.save(
    #                     {
    #                         'model_state': mynet.state_dict(),
    #                         'optimizer': optimizer.state_dict(),
    #                         'transform': transform,
    #                     },
    #                     best_model_path
    #                 )
    #
    #     mynet.eval()
    #     train_x_all = torch.unsqueeze(train_data.data, dim=1).type(torch.FloatTensor)/255
    #     train_y_all = train_data.targets
    #     test_x_all = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255
    #     test_y_all = test_data.targets
    #
    #     train_logits, _ = mynet(train_x_all)
    #     test_logits, _ = mynet(test_x_all)
    #     train_loss_last = criterion(train_logits, train_y_all)
    #     test_loss_last = criterion(test_logits, test_y_all)
    #     pred_train_all = torch.max(train_logits, 1)[1].data.numpy()
    #     pred_test_all = torch.max(test_logits, 1)[1].data.numpy()
    #     acc_train = float((pred_train_all == train_y_all.data.numpy()).astype(int).sum()) / float(train_y_all.size(0))
    #     acc_test = float((pred_test_all == test_y_all.data.numpy()).astype(int).sum()) / float(test_y_all.size(0))
    #     log.logger.info(
    #         'train loss: {:.4f} | test loss: {:.4f} | train acc: {:.2f}% | test acc: {:.2f}%'.format(
    #             train_loss_last, test_loss_last, acc_train*100, acc_test*100
    #         )
    #     )
    #     writer.add_scalar('train/acc', acc_train, total_step)
    #     writer.add_scalar('test/acc', acc_test, total_step)
    # writer.close()

    my_net = CNN()
    params = torch.load(model_path)
    my_net.load_state_dict(params['model_state'])

    # used in my own data

    cnt = 0
    acc_ = 0
    for img, label in my_dataset:
        # plt.subplot(1, 2, 1)
        # plt.imshow(test_x[test_y.eq(label)][0].reshape((28, 28)), plt.gray())
        # plt.title('mnist')
        # plt.subplot(1, 2, 2)
        # plt.imshow(1-img[0], plt.gray())
        # plt.title('mydata')
        # plt.show()
        result, bottleneck = my_net(1-img[0].reshape(1, 1, 28, 28))
        predicted = torch.argmax(result)
        print('predict:{}|real:{}'.format(predicted, label))
        if predicted == label:
            acc_ += 1
        cnt += 1
    acc_ = acc_ * 100.0 / cnt
    log.logger.info('the accuracy of our own handwriting data is {}%'.format(acc_))
    # torch.save(
    #     {
    #         'model_state': params['model_state'],
    #         'optimizer': params['optimizer'],
    #         'transform': transform,
    #     },
    #     best_model_path
    # )
    # torch.save(
    #     {
    #         'model_state': params['model_state'],
    #         'optimizer': params['optimizer'],
    #         'loss': params['loss'],
    #         'epoch': params['epoch'],
    #         'lr': params['lr'],
    #         'wc': params['wc'],
    #         'transform': transform,
    #      },
    #     model_path
    # )
