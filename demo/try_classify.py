# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:00:52 2019

@author: 上上下下左左右右baba
"""

import torch
import torchvision
import torchvision.transforms as transforms  #简化引用

import matplotlib.pyplot as plt
import numpy as np

#==============加载并标准化CIFAR10=======#
#################数据加载与预处理
transform = transforms.Compose(                               #将transforms组合在一起
    [transforms.ToTensor(),                                   #数据处理成[0,1]
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #标准化为范围在[-1,1]之间的张量

#训练集
trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True, 
                                        download=True,
                                        transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=4, 
                                          shuffle=True,
                                          num_workers=0)
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

# 输出图像的函数

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机得到一些训练图片
    
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 打印图片标签
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#===========定义卷积神经网络=======
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
   def __init__(self):
      super(Net,self).__init__()
      self.conv1 = nn.Conv2d(3,6,5)
      self.pool = nn.MaxPool2d(2,2)
      self.conv2 = nn.Conv2d(6,16,5)
      self.fc1 = nn.Linear(16*5*5,120)
      self.fc2 = nn.Linear(120,84)
      self.fc3 = nn.Linear(84,10)
      
   def forward(self,x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1,16*5*5)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      
      return(x)

net = Net()

#===========定义损失函数和优化器=========

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#===========训练网络===============

for epoch in range(2):  #对数据集进行多次循环
   
   running_loss = 0.0
   for i,data in enumerate(trainloader,0):
      #得到输入
      inputs,labels = data
      
      #0参数梯度
      optimizer.zero_grad()
      
      #向前+向后+优化
      outputs = net(inputs)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      
      #打印统计
      running_loss += loss.item()
      if i % 2000 == 1999:   #每2000个小批量打印一次
         print("[%d,%5d] loss: %.3f" %(epoch + 1,i +1 ,running_loss/2000))
         running_loss = 0.0
         
print("Finished Training")

#保存模型

torch.save(net,'MOD.pki')

#dataiter = iter(testloader)
#images,labels = dataiter.next()
#
##输出图片
#imshow(torchvision.utils.make_grid(images))
#print("GroundTruth:"," ".join("%5s" %classes[labels[j]] for j in range(4)))
#
#outputs = net(images)
#
#_, predicted = torch.max(outputs, 1)
#
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
#


