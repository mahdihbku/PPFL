# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:02:29 2021

@author: abdullatif
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
import struct
from Data_process import *
from models import *
from tqdm import tqdm
from FL_funcs import *

import socket
import pickle

def recv_msg(sock, expect_msg_type=None):
    # print('Recieving')
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0])
    return msg

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())

soc = socket.socket()
print("Socket is created.")

soc.connect(("127.0.0.1", 10000))
print("Connected to the server.")




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
        else:
            num_labeled=len(train)//5   
            
###################################################################################               
        labeled, unlabeled = torch.utils.data.random_split(train, [num_labeled, (len(train)//5)*4])
    return labeled, unlabeled     

    
    
training_file = "training_file3"
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

# print(X_train)


n_train = len(X_train)

n_valid = len(X_valid)

n_test = len(X_test)

# The shape of an traffic sign image
# image_shape = X_train[0].shape[:-1]

# Number of unique classes/labels in the dataset.
n_classes = len(set(y_train))
          
print('Start...')

torch.manual_seed(1)


epochs = 10

train_dataset = ProcessDataset(training_file, transform=transforms.ToTensor())
valid_dataset = ProcessDataset(validation_file, transform=transforms.ToTensor())


# traindata_split = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) /  num_users) for _ in range(num_users)])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# train_loaders = [torch.utils.data.DataLoader(x, batch_size=32, shuffle=True) for x in traindata_split]


train_loader = PrepareDataLoader(train_loader, to_device)
valid_loader = PrepareDataLoader(valid_loader, to_device)

# train_loaders = [PrepareDataLoader(train_loaders[i], to_device) for i in range(len(train_loaders))]

criterion = nn.CrossEntropyLoss()



clients_model = BasicNet().to(device)

criterion1 = nn.CrossEntropyLoss()
opt =optim.SGD(clients_model.parameters(), lr=0.001, momentum=0.9) 


global_model = BasicNet().to(device)


losses_training = []
losses_testing = []
acc_training = []
acc_testing = []
    
# participant_index = np.random.permutation(num_clients)[:num_participants]   
loss = 0
#for r in range(num_rounds):

"""Keep local model synchrounized with global model"""
R = 2
count_r = 1
models = []

#while True:
count_r=count_r+1
is_last_round = False
while True:
    MSG = recv_msg(soc, 'Messgage from server ') 
    clients_model.load_state_dict(MSG[1].state_dict())
    train_(epochs, clients_model, criterion1, opt, train_loader, valid_loader)
    msg = ['Msg_from_client', clients_model, is_last_round]
    send_msg(soc, msg)
   
   # if count_r == 2:
    #    break;
    
