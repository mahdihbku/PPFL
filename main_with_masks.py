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

training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"

RAND_BIT_SIZE = 16
rand_low = -100.0
rand_high = +100.0

# num_users = 23
# num_participants = 5
# num_clients= 20
# num_rounds = 50
# epochs = 20
num_clients= 5
data_splits = num_clients
num_participants = num_clients
num_rounds = 3
epochs = 3
LR=0.001
MOMENTUM=0.9
BATCH_SIZE = 32

torch.backends.cudnn.benchmark=True

# to use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

train_dataset = ProcessDataset(training_file, transform=transforms.ToTensor())
valid_dataset = ProcessDataset(validation_file, transform=transforms.ToTensor())
test_dataset = ProcessDataset(testing_file, transform=transforms.ToTensor())

dataset_splits_lengths = [int(len(train_dataset)/data_splits) for _ in range(data_splits)]
dataset_splits_lengths[-1] = len(train_dataset) - sum(dataset_splits_lengths[:-1])
traindata_split = torch.utils.data.random_split(train_dataset, dataset_splits_lengths)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_loaders = [torch.utils.data.DataLoader(x, batch_size=BATCH_SIZE, shuffle=True) for x in traindata_split]

train_loader = PrepareDataLoader(train_loader, to_device)
valid_loader = PrepareDataLoader(valid_loader, to_device)
test_loader = PrepareDataLoader(test_loader, to_device)

train_loaders = [PrepareDataLoader(train_loaders[i], to_device) for i in range(len(train_loaders))]

criterion = nn.CrossEntropyLoss()

clients_models = [BasicNet().to(device) for _ in range(num_clients)] 

criterion1 = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
opt = [optim.SGD(clients_models[i].parameters(), lr=LR, momentum=MOMENTUM) for i in range(num_clients)]

global_model = BasicNet().to(device)

losses_training = []
losses_testing = []
acc_training = []
acc_testing = []
    
participant_index = np.random.permutation(num_clients)[:num_participants]   

for r in range(num_rounds):

    """Keep local model synchrounized with global model"""
    replicate_model(global_model, clients_models)
        
    for i in tqdm(range(num_participants)):
        train_(epochs, clients_models[participant_index[i]], criterion1, opt[participant_index[i]], train_loaders[participant_index[i]], valid_loader)

    print("Aggregating without masking...")
    server_aggregate(global_model, clients_models)
    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)

    """Generating the seeds that will be used for masking (One seed per participant)"""
    seeds = [random.getrandbits(RAND_BIT_SIZE) for i in range(num_participants)]

    # print("Masking clients_models...")
    # for i in range(num_participants):
    #     mask_model(seeds[i], clients_models[i], rand_high, rand_low)
    # print("******* mask has been added")
    # print("Central unmasking clients_models...")
    # for i in range(num_participants):
    #     unmask_model(seeds[i], clients_models[i], rand_high, rand_low)
    # print("******* mask has been removed")
    # server_aggregate(global_model, clients_models)
    # acc, loss = test_(global_model, criterion, test_loader)

    # print("Central with masks list masking clients_models...")
    # for i in range(num_participants):
    #     mask_model(seeds[i], clients_models[i], rand_high, rand_low)
    # print("******* mask has been added")
    # print("Central with masks unmasking clients_models...")
    # masks_shapes = [parm.size() for parm in clients_models[0].parameters()]
    # for i in range(num_participants):
    #     client_masks = generate_masks_from_seed(seeds[i], masks_shapes, rand_high, rand_low)
    #     for parm, mask in zip(clients_models[i].parameters(), client_masks):
    #         parm.data -= mask
    # print("******* mask has been removed")
    # server_aggregate(global_model, clients_models)
    # acc, loss = test_(global_model, criterion, test_loader)

    print("Distributed masking clients_models...")
    masks_shapes = [parm.size() for parm in clients_models[0].parameters()]
    for i in range(num_participants):
        mask_model(seeds[i], clients_models[i], rand_high, rand_low)
    print("******* mask has been added")
    print("Distributed unmasking clients_models...")
    clients_masks = [generate_masks_from_seed(seeds[i], masks_shapes, rand_high, rand_low) for i in range(num_participants)]
    global_mask = sum_list_masks(clients_masks)
    server_aggregate_masked(global_model, clients_models, global_mask)
    print("******* mask has been removed")
    acc, loss = test_(global_model, criterion, test_loader)

    print(f'##########{r+1}-th round')