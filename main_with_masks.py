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
from PPfuncs import *
from utils import *

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
num_clients = 3
data_splits = num_clients
num_participants = num_clients
committee_size = 3
num_rounds = 3
epochs = 3
LR = 0.001
MOMENTUM = 0.9
BATCH_SIZE = 32
NUM_SPLITS = 2

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
    #     mask_model(seeds[i], clients_models[i], rand_low, rand_high)
    # print("******* mask has been added")
    # print("Central unmasking clients_models...")
    # for i in range(num_participants):
    #     unmask_model(seeds[i], clients_models[i], rand_low, rand_high)
    # print("******* mask has been removed")
    # server_aggregate(global_model, clients_models)
    # acc, loss = test_(global_model, criterion, test_loader)

    # print("Central with masks list masking clients_models...")
    # for i in range(num_participants):
    #     mask_model(seeds[i], clients_models[i], rand_low, rand_high)
    # print("******* mask has been added")
    # print("Central with masks unmasking clients_models...")
    # masks_shapes = [parm.size() for parm in clients_models[0].parameters()]
    # for i in range(num_participants):
    #     client_masks = generate_masks_from_seed(seeds[i], masks_shapes, rand_low, rand_high)
    #     for parm, mask in zip(clients_models[i].parameters(), client_masks):
    #         parm.data -= mask
    # print("******* mask has been removed")
    # server_aggregate(global_model, clients_models)
    # acc, loss = test_(global_model, criterion, test_loader)



    # # TEST SPLITTING A TENSOR THEN RECONSTRUCTING IT:
    # t1 = clients_masks[0][0]
    # lt1 = split_additive_mask(t1, 2, rand_low, rand_high)
    # t2 = torch.stack(sum_list_masks(lt1))
    # print("t1.shape={} get_shape(lt1)={} t2.shape={}".format(t1.shape,get_shape(lt1),t2.shape))
    # if not torch.allclose(t1, t2):
    #     raise("#####not torch.allclose(t1, t2)-----------")
    # else:
    #     print("t1 and t2 are the same...")



    print("Distributed masking clients_models...")
    masks_shapes = [parm.size() for parm in clients_models[0].parameters()]
    for i in range(num_participants):
        mask_model(seeds[i], clients_models[i], rand_low, rand_high)
    print("Mask has been added")
    print("Distributed unmasking clients_models...")
    clients_masks = [generate_masks_from_seed(seeds[i], masks_shapes, rand_low, rand_high) for i in range(num_participants)]    # N*P*mask
    # print("get_shape(clients_masks)={}".format(get_shape(clients_masks)))
    print("{} Client masks generated".format(num_participants))

    global_masks = sum_list_masks(clients_masks)
    # print("get_shape(global_masks)={}".format(get_shape(global_masks)))
    
    clients_masks_splits = [split_additive_masks(client_masks, committee_size, rand_low, rand_high) for client_masks in clients_masks]   # N*P*n*mask
    # print("get_shape(clients_masks_splits)={}".format(get_shape(clients_masks_splits)))
    print("Each client masks splitted over {} shares".format(committee_size))

    # A = [[torch.stack(sum_list_masks(P)) for P in N] for N in clients_masks_splits]
    # print("get_shape(A)={}".format(get_shape(A)))
    # if not are_tensors_equal(A, clients_masks):
    #     raise("#####not are_tensors_equal(A[0][0], clients_masks[0][0])-----------")
    # print("A and clients_masks are the same...")
    # global_masks2 = sum_list_masks(A)
    # print("get_shape(global_masks2)={}".format(get_shape(global_masks2)))
    # if not are_tensors_equal(global_masks, global_masks2):
    #     raise("#####not are_tensors_equal(global_masks, global_masks2)-----------")
    # print("global_masks and global_masks2 are the same...")

    transposed_clients_masks_splits = transpose_list(clients_masks_splits, axes=(2,0,1))   # n*N*P*mask
    # print("get_shape(transposed_clients_masks_splits)={}".format(get_shape(transposed_clients_masks_splits)))
    print("Client masks transposed")

    if not are_tensors_equal(clients_masks_splits, transpose_list(transposed_clients_masks_splits, axes=(1,2,0))):
        raise("#####transpose_list tests NOT successful-----------")

    global_masks_splits = [sum_list_masks(one_transposed_clients_masks_splits) for one_transposed_clients_masks_splits in transposed_clients_masks_splits]
    # print("get_shape(global_masks_splits)={}".format(get_shape(global_masks_splits)))
    print("Global masks splits calculated")

    global_masks3 = sum_list_masks(global_masks_splits)
    # print("get_shape(global_masks3)={}".format(get_shape(global_masks3)))
    print("Global masks calculated")

    if not are_tensors_equal(global_masks, global_masks3):
        raise("#####not are_tensors_equal(global_masks, global_masks3)!!!")

    server_aggregate_masked(global_model, clients_models, global_masks)
    print("Mask removed and global model aggregated successfully")
    acc, loss = test_(global_model, criterion, test_loader)

    print(f'##########{r+1}-th round')
