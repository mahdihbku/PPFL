# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Mahdi 
"""

import socket
import pickle
from utils import *
from PPfuncs import *
from models import BasicNet
import sys
from params import *
import secrets
import ecvrf_edwards25519_sha512_elligator2

if (len(sys.argv) != 2):
    raise "You should give the listening port number of this committee member in the command line as: python PPcommitteeMember.py PORT"
MY_PORT = int(sys.argv[1])

secret_key = secrets.token_bytes(nbytes=32)
public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(("127.0.0.1", MY_PORT))
print("Listening socket is bound to address 127.0.0.1 & port {}".format(MY_PORT))
server_soc = socket.socket()
server_soc.connect((SERVER_IP, SERVER_PORT_FOR_COMMITTEE))
send_msg(server_soc, ['Committee member info', str(MY_PORT), public_key])
print("Connected to the server")
print("Waiting for committee list from the server...")
msg0 = recv_msg(server_soc, 'Server committee list message')
committee_list = msg0[1]

p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, pickle.dumps(committee_list))
b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
is_qualified = count_leading_zeros(beta_string) >= leading_bits

# print(f"Secret Key Size: {len(secret_key)} bytes")
# print(f"Public Key Size: {len(public_key)} bytes")
# print(f"Proof Size: {len(pi_string)} bytes")

if is_qualified:
    print("Qualified for the upcoming rounds, sending the proof to the server...")
    send_msg(server_soc, ['Committee member qualification message', pi_string])
else:
    send_msg(server_soc, ['Committee member qualification message', 'not qualified']) # just to avoid timeout
    print("Not qualified for the upcoming rounds, quitting...")
    sys.exit()

basic_model = BasicNet()

round = 1
is_last_round = False
while True:
    print("########## Round {}".format(round))
    print("Waiting for clients to connect...")
    partcicpants=[]
    while len(partcicpants) < num_participants:
        listening_sock.listen(MAX_CONNECTIONS)
        (participant_sock, (ip, port)) = listening_sock.accept()
        partcicpants.append(participant_sock)
    print('All clients have joined')
    print("Waiting for clients to send their masks...")
    clients_masks_splits = []
    for i in range(num_participants):   # TODO should run in parallel...
        msg0 = recv_msg(partcicpants[i], 'Client mask split message')
        clients_masks_splits.append(msg0[1])
        is_last_round = msg0[2]
    print("All masks splits received from clients")
    local_global_mask = sum_list_masks(clients_masks_splits)
    print("Sending the global mask to the server...")
    msg = ['Committee mask message', local_global_mask, False]
    send_msg(server_soc, msg)
    print("Global mask sent to server")
    if is_last_round:
        print("Last round reached, quitting...")
        break
    round += 1
