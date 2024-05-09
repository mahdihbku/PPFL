# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 00:39:50 2021

@author: Windows
"""

# -*- coding: utf-8 -*-
"""
Created on  Dec  3

@author: abdullatif 
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:27:31 2020

@author: abdullatif
"""

# import here
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F
from Data_process import *
from models import *
from tqdm import tqdm
from FL_funcs import *
torch.backends.cudnn.benchmark=True

# to use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""This function is to prepare the unlabeled and labeled dataset for each user"""
def set_unlabeled(data,dataset,total_users,unlabeled_precent=0.7):
    unlabeled_precent=unlabeled_precent
    for i in range(total_users):
        id, train , test = read_user_data(i, data, dataset)
        #### Split the data into lableled and unlabeled data samples
        if (len(train))%5>0: 
#### If the number of data samples is odd then add 1 to avoid the completion error
            num_labeled=len(train)//5 + len(train) %5
            # // floor devision take the int only
        else:
            num_labeled=len(train)//5   
            
###################################################################################               
        labeled, unlabeled = torch.utils.data.random_split(train, [num_labeled, (len(train)//5)*4])
    return labeled, unlabeled     

    
    
training_file = "data/train"
validation_file = "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print(len(y_train))

Num_clients= 20# we have 23
name ='training_file'
user = {}

for i in range(Num_clients):
    #['features'] 
    user['features'] = train['features'][i*1500:(i+1)*1500]
    user['labels'] = train['labels'][i*1500:(i+1)*1500]
    name_file_user = name +str(i+1)
    # step1 open the file we want to write the data in 
    outfile = open(name_file_user, 'wb') 
    # step2 use pickle.dump() to write data 
    pickle.dump(user,outfile)
    outfile.close()