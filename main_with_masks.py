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
from PP_funcs import *
torch.backends.cudnn.benchmark=True

# to use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
training_file = "data/train.p"
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
# num_users = 23
# num_participants = 5
# num_clients= 20
# num_rounds = 50
# epochs = 20

num_users = 23
num_participants = 5
num_clients= 5
num_rounds = 3
epochs = 5

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
criterion = nn.CrossEntropyLoss()
opt = [optim.SGD(clients_models[i].parameters(), lr=0.001, momentum=0.9) for i in range(num_clients)]

global_model = BasicNet().to(device)


losses_training = []
losses_testing = []
acc_training = []
acc_testing = []
    
participant_index = np.random.permutation(num_clients)[:num_participants]   
loss = 0
bit_size = 16
low = -100.0
high = +100.0

for r in range(num_rounds):

    """Keep local model synchrounized with global model"""
    replicate_model(global_model, clients_models)
        
    for i in tqdm(range(num_participants)):
        train_(epochs, clients_models[participant_index[i]], criterion1, opt[participant_index[i]], train_loaders[participant_index[i]], valid_loader)

    """Generating the seeds that will be used for masking"""
    seeds = []
    for i in range(num_participants):
        seeds.append(random.getrandbits(bit_size))

    print("Aggregating without masking...")
    server_aggregate(global_model, clients_models)
    acc, loss = test_(global_model, criterion, test_loader)

    print("Masking clients_models...")
    for i in range(num_participants):
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        for parm in clients_models[i].parameters():
            # print("clients_models[0]: param name: {}".format(name))
            # print("clients_models[0]: param.size(): {}".format(parm.size()))
            # print("clients_models[0]: param param.requires_grad: {}".format(parm.requires_grad))
            mask = torch.rand(parm.size()) * (high - low) + low # One mask for each parm
            # print("clients_models[{}]: param name: {}".format(i,name))
            # print("type(mask): {}".format(type(mask)))
            # print("type(parm): {}".format(type(parm)))
            # print("type(parm.data): {}".format(type(parm.data)))
            parm.data += mask
    print("******* mask has been added")

    print("Central unmasking clients_models...")
    for i in range(num_participants):
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        for parm in clients_models[i].parameters():
            mask = torch.rand(parm.size()) * (high - low) + low
            parm.data -= mask
    print("******* mask has been removed")
    server_aggregate(global_model, clients_models)
    acc, loss = test_(global_model, criterion, test_loader)


    print("Central with masks list masking clients_models...")
    for i in range(num_participants):
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        for parm in clients_models[i].parameters():
            mask = torch.rand(parm.size()) * (high - low) + low # One mask for each parm
            parm.data += mask
    print("******* mask has been added")

    print("Central with masks unmasking clients_models...")
    masks_shapes = [parm.size() for parm in clients_models[0].parameters()]
    for i in range(num_participants):
        client_masks = generate_masks_from_seed(seeds[i], masks_shapes, high, low)
        for parm, mask in zip(clients_models[i].parameters(), client_masks):
            parm.data -= mask
    print("******* mask has been removed")
    server_aggregate(global_model, clients_models)
    acc, loss = test_(global_model, criterion, test_loader)

    print("Distributed masking clients_models...")
    for i in range(num_participants):
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        for parm in clients_models[i].parameters():
            mask = torch.rand(parm.size()) * (high - low) + low # One mask for each parm
            parm.data += mask
    print("******* mask has been added")

    print("Distributed unmasking clients_models...")
    # Compute global mask
    clients_masks = []
    for i in range(num_participants):
        client_masks = generate_masks_from_seed(seeds[i], masks_shapes, high, low)
        clients_masks.append(client_masks)

    global_mask = sum_list_masks(clients_masks)
    
    server_aggregate_masked(global_model, clients_models, global_mask)

    print("******* mask has been removed")
    acc, loss = test_(global_model, criterion, test_loader)

    # replicate_model(global_model, clients_models)

    print(f'##########{r+1}-th round')
    # """Evalute The global model every global round"""
    # acc, loss = test_(global_model, criterion, test_loader)
    # losses_testing.append(loss)
    # acc_testing.append(acc)
    
    # acc, loss = test_(global_model2, criterion, test_loader)
    # losses_testing.append(loss)
    # acc_testing.append(acc)