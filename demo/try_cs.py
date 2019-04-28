# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 21:28:46 2019

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

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))