# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 03:05:34 2021

@edit  maymouna 
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
# import cv2
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

    
   # user 1 data  
training_file = "training_file1.p"
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
          
print('Start...')

torch.manual_seed(1)
num_users = 23
num_participants = 5
num_clients= 20
num_rounds = 50
epochs = 20

train_dataset = ProcessDataset(training_file, transform=transforms.ToTensor())
valid_dataset = ProcessDataset(validation_file, transform=transforms.ToTensor())
test_dataset = ProcessDataset(testing_file, transform=transforms.ToTensor())

traindata_split = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) /  num_users) for _ in range(num_users)])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_loaders = [torch.utils.data.DataLoader(x, batch_size=32, shuffle=True) for x in traindata_split]


train_loader = PrepareDataLoader(train_loader, to_device)
valid_loader = PrepareDataLoader(valid_loader, to_device)
test_loader = PrepareDataLoader(test_loader, to_device)


train_loaders = [PrepareDataLoader(train_loaders[i], to_device) for i in range(len(train_loaders))]

criterion = nn.CrossEntropyLoss()



clients_models = [BasicNet().to(device) for _ in range(num_clients)] 

criterion1 = nn.CrossEntropyLoss()
opt = [optim.SGD(clients_models[i].parameters(), lr=0.001, momentum=0.9) for i in range(num_clients)]


global_model = BasicNet().to(device)


losses_training = []
losses_testing = []
acc_training = []
acc_testing = []
    
participant_index = np.random.permutation(num_clients)[:num_participants]   
loss = 0
for r in range(num_rounds):

    for model in clients_models:
        """Keep local model synchrounized with global model"""
        model.load_state_dict(global_model.state_dict())
        
    for i in tqdm(range(num_participants)):
        train_(epochs, clients_models[participant_index[i]], criterion1, opt[participant_index[i]], train_loaders[participant_index[i]], valid_loader)
    
    """Local Models aggregations and Global model Fusion"""
    server_aggregate(global_model, clients_models)   
    print(f'##########{r+1}-th round')
    """Evalute The global model every global round"""
    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)
    