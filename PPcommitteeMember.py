# -*- coding: utf-8 -*-
"""
Created on  May 8 2024

@author: Mahdi 
"""

import socket
import pickle
from utils import *
from PP_funcs import *
from models import BasicNet

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(("127.0.0.1", COMMITTEE_PORT))
print("Socket is bound to address localhost & port {}".format(COMMITTEE_PORT))
server_soc = socket.socket()
server_soc.connect((SERVER_IP, SERVER_PORT_FOR_COMMITTEE))

basic_model = BasicNet()

round = 1
while True:
    print("########## Round {}".format(round))
    print("Waiting for clients to connect...")
    partcicpants=[]
    while len(partcicpants) < num_participants:
        listening_sock.listen(MAX_CONNECTIONS)
        (participant_sock, (ip, port)) = listening_sock.accept()
        partcicpants.append(participant_sock)
    print('All clients have joined')
    print("Waiting for clients to send their random seeds...")
    seeds = []
    for i in range(num_participants):   # TODO should run in parallel...
        msg0 = recv_msg(partcicpants[i], 'Messgage from client ')  
        seeds.append(msg0[1])
    print("All seeds received from clients")
    masks_shapes = [parm.size() for parm in basic_model.parameters()]
    clients_masks = [generate_masks_from_seed(seeds[i], masks_shapes, rand_high, rand_low) for i in range(num_participants)]
    global_mask = sum_list_masks(clients_masks)
    print("Sending the global mask to the server...")
    msg = ['Msg_from_committee', global_mask, False]
    send_msg(server_soc, msg)
    print("Global mask sent to server")
    round += 1
