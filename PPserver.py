# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:49:47 2021

@author: abdullatif
"""

import socket
import pickle
from Data_process import *
from models import *
from tqdm import tqdm
from FL_funcs import *
from PPfuncs import sum_list_masks
from utils import *
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from params import *

clients_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clients_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clients_sock.bind((SERVER_IP, SERVER_PORT_FOR_CLIENT))
print("Clients socket is bound to address {} & port {}".format(SERVER_IP, SERVER_PORT_FOR_CLIENT))

committee_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
committee_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
committee_sock.bind((SERVER_IP, SERVER_PORT_FOR_COMMITTEE))
print("Committee socket is bound to address {} & port {}".format(SERVER_IP, SERVER_PORT_FOR_COMMITTEE))

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
participants = []
while len(participants) < num_participants:
    clients_sock.listen(MAX_CONNECTIONS)
    (participant_sock, (ip, port)) = clients_sock.accept()
    print('New connection from client ', (ip,port))
    participants.append(participant_sock)
print('All participants have joined')

print("Waiting for {} committee members to join...".format(committee_size))
committee_members = []
while len(committee_members) < committee_size:
    committee_sock.listen(MAX_CONNECTIONS)
    (committee_member_sock, (ip, port)) = committee_sock.accept()
    print('New connection from committee member ', (ip,port))
    msg0 = recv_msg(committee_member_sock, 'Messgage from committee ')
    committee_members.append({'sock':committee_member_sock, 'ip':ip, 'port':int(msg0[1])})
print('All committee members have joined')

is_last_round = True if num_rounds == 1 else False
losses_testing = []
acc_testing = []
committee_list = [{'ip':cm['ip'],'port':cm['port']} for cm in committee_members]
print("committee_list: {}".format(committee_list))
print("Sending the global model to clients...")
msg = ['MSG_SERVER_TO_CLIENT_INTILAIZATION', global_model, is_last_round, committee_list]
for participant in participants:
    send_msg(participant, msg)

round = 1
while True:
    print("########## Round {}".format(round))

    print("Waiting for local models from clients...")
    clients_models = []
    for i in range(num_participants):   # TODO should run in parallel...
        msg0 = recv_msg(participants[i], 'Messgage from client ')  
        clients_models.append(msg0[1])
    print("Local models from clients received")

    print("Waiting for masks from committee members...")
    global_masks_splits = []
    for i in range(committee_size):   # TODO should run in parallel...
        msg0 = recv_msg(committee_members[i]['sock'], 'Messgage from committee ')  
        global_masks_splits.append(msg0[1])
    global_masks = sum_list_masks(global_masks_splits)
    print("Global mask computed")

    # for param in clients_models[0].parameters():
    #     print("before removing masks my_model.parameters()={}".format(param.data))
    
    server_aggregate_masked(global_model, clients_models, global_masks)

    # for param in global_model.parameters():
    #     print("new global model global_model.parameters()={}".format(param.data))
    
    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)

    if round+1 >= num_rounds:
        is_last_round = True

    msg = ['Server_Sends_GM', global_model, is_last_round]
    for p in participants:
        send_msg(p, msg)

    if round == num_rounds:
        print("Last round reached, quitting...")
        break
    round += 1

# print('losses_testing',losses_testing)
# print('acc_testing',acc_testing)
