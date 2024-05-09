# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:49:47 2021

@author: Windows
"""

import socket
import pickle
import struct
listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(("192.168.43.47", 10000))
print("Socket is bound to an address & port number.")

partcicpants=[]
num_selected = 1

print("Listening for incoming connection ...")

def recv_msg(sock, expect_msg_type=None):
    print('Start Revieving')
    msg_len = struct.unpack(">I", sock.recv(1))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0])
print("Socket is closed.")

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())


while len(partcicpants) < num_selected:
    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (participant_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    print(participant_sock)
    partcicpants.append(participant_sock)
# Establish connections to each client, up to n_nodes clients
send_msg(partcicpants[0], msg)

recv_msg(partcicpants[0], 'MSG_DATA_PREP_FINISHED_CLIENT_TO_SERVER')    