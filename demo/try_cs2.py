# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:45:46 2019

@author: 上上下下左左右右baba
"""

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(                               #将transforms组合在一起
    [transforms.ToTensor(),                                   #数据处理成[0,1]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #标准化为范围在[-1,1]之间的张量

#测试集
testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True, 
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=4, 
                                         shuffle=False,       #？  为什么是false
                                         num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

the_model = torch.load('MOD.pki')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))