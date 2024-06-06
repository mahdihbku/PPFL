# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:02:29 2021

@author: abdullatif
"""

# import here
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
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
import sys
from FL_funcs import *
from PPfuncs import *
from utils import *
from params import *
import secrets
import ecvrf_edwards25519_sha512_elligator2

import socket
import pickle

if (len(sys.argv) != 2):
    raise "You should give the client index in the command line as: python PPclient.py INDEX"
MY_CLIENT_INDEX = sys.argv[1]

server_soc = socket.socket()
server_soc.connect((SERVER_IP, SERVER_PORT_FOR_CLIENT))
print("Connected to the server")

# Generating a public and private key pair
secret_key = secrets.token_bytes(nbytes=32)
public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)

send_msg(server_soc, ['Client info', public_key])
print("Public key sent to the server")

# Initialize the training environment
torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
training_file = training_files_location + "training_file" + str(MY_CLIENT_INDEX)
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
n_train = len(X_train)
n_valid = len(X_valid)

print('Start...')

torch.manual_seed(1)
train_dataset = ProcessDataset(training_file, transform=transforms.ToTensor())
valid_dataset = ProcessDataset(validation_file, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_loader = PrepareDataLoader(train_loader, to_device)
valid_loader = PrepareDataLoader(valid_loader, to_device)
my_model = BasicNet().to(device)
criterion = nn.CrossEntropyLoss()
opt =optim.SGD(my_model.parameters(), lr=LR, momentum=MOMENTUM) 

round = 1
while True:
    print("Waiting for the global model from the server...")
    msg0 = recv_msg(server_soc, 'Server global model message') 
    my_model.load_state_dict(msg0[1].state_dict())
    is_last_round = msg0[2]
    if (round == 1):
        committee_list = msg0[3]    # n * (ip, port, key)
        selected_committe_members_list = msg0[4]    # C * (index, proof)
        print("Current committee members:{}".format(committee_list))
        print("Current selected committee members:{}".format(selected_committe_members_list))
    print("########## Round {}".format(round))
    committee_sockets = []
    for i in range(len(selected_committe_members_list)):
        cm_soc = socket.socket()
        cm_soc.connect((committee_list[selected_committe_members_list[i]['index']]['ip'], committee_list[selected_committe_members_list[i]['index']]['port']))
        committee_sockets.append(cm_soc)
    print("Connected to all committee members")
    train_(EPOCHS, my_model, criterion, opt, train_loader, valid_loader)
    acc, loss = test_(my_model, criterion, valid_loader)
    seed = random.getrandbits(RAND_BIT_SIZE)
    masks_shapes = [parm.size() for parm in my_model.parameters()]
    masks = generate_masks_from_seed(seed, masks_shapes, rand_low, rand_high)

    mask_model(seed, my_model, rand_low, rand_high)

    send_msg(server_soc, ['Client local model message', my_model, is_last_round])
    print("Masked local model sent to server")
    masks_splits = split_additive_masks(masks, len(committee_sockets), rand_low, rand_high) # shape: P*n*mask
    masks_splits = transpose_list(masks_splits, axes=(1,0)) # shape: n*P*mask
    for i in range(len(committee_sockets)):    # TODO Should run in parallel
        send_msg(committee_sockets[i], ['Client mask split message', masks_splits[i], is_last_round])
        print("Random mask split sent to committee member {}".format(committee_sockets[i].getpeername()))
        committee_sockets[i].close()
    print("masks sent to all committee members")
    if is_last_round:
        print("Last round reached, quitting...")
        break
    round += 1
