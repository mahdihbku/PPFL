# -*- coding: utf-8 -*-
"""
Created on  Dec  1

@author: abdullatif 
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 00:41:11 2020

@author: abdullatif
"""

from torch import nn, optim
import torch.nn.functional as F


""" Basic model for traffic sign data """
class BasicNet(nn.Module):
    def __init__(self, gray=False):
        super(BasicNet, self).__init__()
        input_chan = 1 if gray else 3
        self.conv1 = nn.Conv2d(input_chan, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x