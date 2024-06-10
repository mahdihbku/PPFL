# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:49:47 2021

@author: abdullatif
"""

import socket
import pickle
import time
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
import ecvrf_edwards25519_sha512_elligator2

report_time = True
report_time_file = "experiments/online_comp/server.csv"

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

# print(f"Total number of parameters: {sum(p.numel() for p in global_model.parameters())}")

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
test_dataset = ProcessDataset(testing_file, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = PrepareDataLoader(test_loader, to_device)

print("Waiting for {} clients to join...".format(num_participants))
list_of_participants = []   # includes the list of all users. N * (socket, key)
while len(list_of_participants) < num_participants:
    clients_sock.listen(MAX_CONNECTIONS)
    (participant_sock, (ip, port)) = clients_sock.accept()
    print('New connection from client ', (ip,port))
    msg0 = recv_msg(participant_sock, 'Client info')
    list_of_participants.append({'socket':participant_sock, 'key':msg0[1]})
print('All participants have joined')

print("Waiting for {} committee members to join...".format(committee_size))
committee_members_list = [] # this list is local; it includes sockets. n * (socket, ip, port, key)
while len(committee_members_list) < committee_size:
    committee_sock.listen(MAX_CONNECTIONS)
    (committee_member_sock, (ip, port)) = committee_sock.accept()
    print('New connection from committee member ', (ip,port))
    msg0 = recv_msg(committee_member_sock, 'Committee member info')
    committee_members_list.append({'sock':committee_member_sock, 'ip':ip, 'port':int(msg0[1]), 'key':msg0[2]})
print('All committee members have joined')

is_last_round = True if num_rounds == 1 else False
losses_testing = []
acc_testing = []
committee_participants_list = [{'ip':cm['ip'],'port':cm['port'],'key':cm['key']} for cm in committee_members_list] # this list is global; will be shared with all users. n * (ip, port, key)
print("participants_list: {}".format(committee_participants_list))

print("Sending committee participants list to committee members...")
msg = ['Server committee participants list message', committee_participants_list]
for cm in committee_members_list:
    send_msg(cm['sock'], msg)

selected_committe_members_list = [] # this list will be shared with all users. C * (index, proof)
for i in range(len(committee_members_list)):
    msg0 = recv_msg(committee_members_list[i]['sock'], 'Committee member qualification message')
    if msg0[1] != 'not qualified':
        '''Verify the proof of qualification'''
        _, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(committee_members_list[i]['key'], msg0[1], pickle.dumps(committee_participants_list))
        is_qualified = count_leading_zeros(beta_string) >= leading_bits
        if is_qualified:
            selected_committe_members_list.append({'index':i,'proof':msg0[1]})
        else:
            print("The proof received from committee member {} ({}:{}) is not valid!".format(i, committee_members_list[i]['ip'], committee_members_list[i]['port']))
print("Number if elected committee members for the upcoming rounds is {}/{}".format(len(selected_committe_members_list), len(committee_participants_list)))

print("Sending the global model to clients...")
msg = ['Server global model message', global_model, is_last_round, committee_participants_list, selected_committe_members_list]
for participant in list_of_participants:
    send_msg(participant['socket'], msg)

round = 1
while True:
    print("########## Round {}".format(round))

    print("Waiting for local models from clients...")
    clients_models = []
    for i in range(num_participants):   # TODO should run in parallel...
        msg0 = recv_msg(list_of_participants[i]['socket'], 'Client local model message')
        clients_models.append(msg0[1])
    print("Local models from clients received")

    print("Waiting for masks from committee members...")
    global_masks_splits = []
    for i in range(len(selected_committe_members_list)):   # TODO should run in parallel...
        msg0 = recv_msg(committee_members_list[selected_committe_members_list[i]['index']]['sock'], 'Committee mask message')
        global_masks_splits.append(msg0[1])

    t1 = time.perf_counter()
    global_masks = sum_list_masks(global_masks_splits)
    server_aggregate_masked(global_model, clients_models, global_masks)
    comp_t = (time.perf_counter() - t1)*1000

    if report_time:
        N = num_participants
        C = len(selected_committe_members_list)
        P = sum(p.numel() for p in global_model.parameters())
        f = open(report_time_file, "a")
        f.write(str(N)+', '+str(C)+', '+str(P)+', '+str(comp_t)+'\n')
        f.close()
        print("Reported time saved in file {}".format(report_time_file))

    print("Global model unmasked")

    acc, loss = test_(global_model, criterion, test_loader)
    losses_testing.append(loss)
    acc_testing.append(acc)

    if round+1 >= num_rounds:
        is_last_round = True

    msg = ['Server global model message', global_model, is_last_round]
    for p in list_of_participants:
        send_msg(p['socket'], msg)

    if round == num_rounds:
        print("Last round reached, quitting...")
        break
    round += 1

# print('losses_testing',losses_testing)
# print('acc_testing',acc_testing)
