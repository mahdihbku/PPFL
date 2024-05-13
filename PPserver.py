# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:49:47 2021

@author: Windows
"""

import socket
import pickle
import struct
from Data_process import *
from models import *
from tqdm import tqdm
from FL_funcs import *
from utils import *
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind((SERVER_IP, SERVER_PORT))
print("Socket is bound to an address & port number.")

partcicpants=[]

device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
global_model = BasicNet().to(device)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
test_dataset = ProcessDataset(testing_file, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = PrepareDataLoader(test_loader, to_device)

print("Waiting for {} clients to join...".format(num_participants))

partcicpants = []
while len(partcicpants) < num_participants:
    listening_sock.listen(MAX_CONNECTIONS)
    (participant_sock, (ip, port)) = listening_sock.accept()
    print('New connection from ', (ip,port))
    partcicpants.append(participant_sock)
print('All participants have joined')

is_last_round = False
losses_testing = []
acc_testing = []
print("Sending the global model to the clients...")
msg = ['MSG_SERVER_TO_CLIENT_INTILAIZATION', global_model, is_last_round]
for k in range(num_participants):
    send_msg(partcicpants[k], msg)

round = 1
while True:
    print("########## Round {}".format(round))

    print("Waiting for local models from the clients...")
    models = []
    for count_participant in range(num_participants):
        msg0 = recv_msg(partcicpants[count_participant], 'Messgage from client ')  
        models.append(msg0[1])

    server_aggregate(global_model, models)
    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)

    if round+1 == num_rounds:
        is_last_round = True

    msg = ['Server_Sends_GM', global_model, is_last_round]
    for k in range(num_participants):
        send_msg(partcicpants[k], msg)

    if round == num_rounds:  
        break
    round += 1

# print('losses_testing',losses_testing)
# print('acc_testing',acc_testing)
