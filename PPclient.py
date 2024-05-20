# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:02:29 2021

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
import struct
from Data_process import *
from models import *
from tqdm import tqdm
import sys
from FL_funcs import *
from PP_funcs import *
from utils import *

import socket
import pickle

if (len(sys.argv) != 2):
    raise "You should give the client index in the command line as: python PPclient.py INDEX"
MY_CLIENT_INDEX = sys.argv[1]

server_soc = socket.socket()
server_soc.connect((SERVER_IP, SERVER_PORT_FOR_CLIENT))
print("Connected to the server.")

torch.backends.cudnn.benchmark=True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
training_file = training_files_location + "training_file" + str(MY_CLIENT_INDEX)

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)
n_valid = len(X_valid)
n_test = len(X_test)

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
    msg0 = recv_msg(server_soc, 'Messgage from server ') 
    my_model.load_state_dict(msg0[1].state_dict())
    is_last_round = msg0[2]
    if (round == 1):
        committee_list = msg0[3]
    print("########## Round {}".format(round))
    for cm in committee_list:
        ip_address, port = cm
        cm_soc = socket.socket()
        cm_soc.connect((ip_address, COMMITTEE_PORT))    # TODO COMMITTEE_PORT or port ?!?!?!
    train_(epochs, my_model, criterion, opt, train_loader, valid_loader)
    seed = random.getrandbits(RAND_BIT_SIZE)
    mask_model(seed, my_model, rand_high, rand_low)
    send_msg(server_soc, ['Msg_from_client', my_model, is_last_round])
    print("Masked local model sent to server")
    for cm in committee_list:
        send_msg(cm_soc, ['Msg_from_client', seed])
        cm_soc.close()
        print("Random seed sent to committee member {}".format(cm))
    print("Seed sent to committee members")
    round += 1
    if is_last_round:
        break
