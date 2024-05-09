# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:33:51 2021

@author: Windows
"""

# make a connection beween the clinet and the server your laptop is the server 

import socket

HOST = '10.40.21.181'  # Standard loopback interface address (localhost)
PORT = 10000       # Port to listen on (non-privileged ports are > 1023)
# the port number
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)