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
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(("192.168.43.47", 10000))
print("Socket is bound to an address & port number.")
testing_file = "data/test.p"

partcicpants=[]

num_selected = 2

device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
global_model = BasicNet().to(device)

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
test_dataset = ProcessDataset(testing_file, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loader = PrepareDataLoader(test_loader, to_device)

print("Listening for incoming connection ...")

def recv_msg(sock, expect_msg_type=None):
    print('Receiving an update')
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0])
    return msg
print("Socket is closed.")

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())

partcicpants = []
while len(partcicpants) < num_selected:
    listening_sock.listen(5)
    print("Waiting for participants to join...")
    (participant_sock, (ip, port)) = listening_sock.accept()
    print('New connection from ', (ip,port))
    print(participant_sock)
    partcicpants.append(participant_sock)
# Establish connections to each client, up to n_nodes clients
is_last_round = True
losses_testing=[]
acc_testing=[]
msg = ['MSG_SERVER_TO_CLIENT_INTILAIZATION', global_model, is_last_round]
for k in range(num_selected):
    send_msg(partcicpants[k], msg)
print('All Participants Joined')
r =1
R = 50
while True:
    print('###################### Start Learning ####################################')
    models =[]
    for count_participant in range(num_selected):
        # print("Local Models aggregations and Global model Fusion")
        msg0= recv_msg(partcicpants[count_participant], 'Messgage from client ')  
        models.append(msg0[1])
    server_aggregate(global_model, models)
    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)
 
    msg = ['Server_Sends_GM', global_model, is_last_round]
    for k in range(num_selected):
        send_msg(partcicpants[k], msg)
    r = r+1
    if r == R:
        is_last_round = True  
        break 
print('losses_testing',losses_testing)
print('acc_testing',acc_testing)

