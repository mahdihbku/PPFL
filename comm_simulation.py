# -*- coding: utf-8 -*-
"""
Created on Thu Mai 29 2024

@author: Mahdi
"""

# Communication cost
'''
    Client                                  Server                              Committee
Offline:
            --connect-->
            N*--pub key: key_size-->
            N*<--clients_list: N*(pub key,info): N*(key_size+user_info_size)--

Online:
                N*<--GM: P*param_size--
                                                    <--selection proof: C*proof_size--
                N*<--committee_list: C*proof_size--
                N*--masked_LM: P*param_size-->
                N*--------------------------------------------masks: (C-1)*seed+mask: (C-1)seed_size+P*rand_number_size-->
                                                                C*<--masks: P*rand_number_size--

Baseline:
                N*<--GM: P*param_size--
                N*--masked_LM: P*param_size-->
'''

from torch import seed

key_size = 32   # bytes
user_info_size = 20 # bytes
param_size = 16 # bytes (long double)
proof_size = 80 # bytes
rand_number_size = 16   # bytes
seed_size = 16 # bytes
Ps = [10000, 64811, 100000, 1000000, 10000000, 11511784, 25557032, 100000000, 138357544]    # simple model for traffic flow, ResNet-18, ResNet-50, VGG16
Cs = [3, 5, 10, 15, 20]
Ns = [10, 100, 200, 300, 400, 500]

# Offline communication
off_client_comm_list = []
off_server_comm_list = []
for N in Ns:
    off_client_comm_list.append((key_size + N*(key_size+user_info_size))/1024)
    off_server_comm_list.append((N*key_size + N*N*(key_size+user_info_size))/1024)

# Online communication
# For 1 client
client_comm_list = []
server_comm_list = []
client_comm_baseline_list = []
server_comm_baseline_list = []
committee_comm_list = []
for N in Ns:
    for C in Cs:
        for P in Ps:
            client_comm_list.append((P*param_size + C*proof_size + P*param_size + (C-1)*seed_size + P*rand_number_size)/1024/1024)
            server_comm_list.append((N*P*param_size + N*C*proof_size + N*P*param_size + C*P*rand_number_size)/1024/1024)
            # committee_comm_list.append((N*(((C-1)*seed_size+P*rand_number_size))/C + P*rand_number_size)/1024/1024)
            committee_comm_list.append((N*(((C-1)*seed_size+P*rand_number_size))/C)/1024/1024)
            client_comm_baseline_list.append(2*P*param_size/1024/1024)
            server_comm_baseline_list.append(2*N*P*param_size/1024/1024)

f = open("experiments/offline_comm/offline_comm.csv", "w")
for i in range(len(Ns)):
    f.write(str(Ns[i])+', '+str(off_client_comm_list[i])+', '+str(off_server_comm_list[i])+'\n')
f.close()

f = open("experiments/online_comm/online_comm.csv", "w")
i = 0
for N in Ns:
    for C in Cs:
        for P in Ps:
            f.write(str(N)+', '+str(C)+', '+str(P)+', '+str(client_comm_list[i])+', '+str(server_comm_list[i])+', '+str(committee_comm_list[i])+', '+str(client_comm_baseline_list[i])+', '+str(server_comm_baseline_list[i])+'\n')
            i += 1
f.close()
