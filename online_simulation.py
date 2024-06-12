# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:02:29 2021

@author: Mahdi
"""

import time
from utils import *
from PPfuncs import *
from FL_funcs import *
from models import BasicNet
from params import *

Ns = [10, 100, 200, 500, 700, 1000]
Cs = [3, 5, 10, 20, 30]

device = torch.device("cpu")
global_model = BasicNet().to(device)
my_model = BasicNet().to(device)

f = open("experiments/online_comp/online_sim.csv", "w")
for N in Ns:
    for C in Cs:
        t1 = time.perf_counter()
        seed = random.getrandbits(RAND_BIT_SIZE)
        masks_shapes = [parm.size() for parm in my_model.parameters()]
        masks = generate_masks_from_seed(seed, masks_shapes, rand_low, rand_high)
        mask_model(seed, my_model, rand_low, rand_high)
        masks_splits = split_additive_masks(masks, C, rand_low, rand_high) # shape: P*n*mask
        masks_splits = transpose_list(masks_splits, axes=(1,0)) # shape: n*P*mask
        client_t = (time.perf_counter() - t1)*1000

        # committee member time
        seeds = [random.getrandbits(RAND_BIT_SIZE) for i in range(N)]
        clients_masks = [generate_masks_from_seed(seeds[i], masks_shapes, rand_low, rand_high) for i in range(N)]    # N*P*mask
        clients_masks_splits = [split_additive_masks(client_masks, committee_size, rand_low, rand_high) for client_masks in clients_masks]   # N*P*n*mask
        transposed_clients_masks_splits = transpose_list(clients_masks_splits, axes=(2,0,1))   # n*N*P*mask
        t1 = time.perf_counter()
        local_global_mask = sum_list_masks(transposed_clients_masks_splits[0])
        com_t = (time.perf_counter() - t1)*1000

        # server time
        clients_models = [BasicNet().to(device) for _ in range(N)]
        replicate_model(global_model, clients_models)
        global_masks_splits = [sum_list_masks(one_transposed_clients_masks_splits) for one_transposed_clients_masks_splits in transposed_clients_masks_splits]
        t1 = time.perf_counter()
        global_masks = sum_list_masks(global_masks_splits)
        server_aggregate_masked(global_model, clients_models, global_masks)
        server_t = (time.perf_counter() - t1)*1000

        print("N={} C={} client_t={} com_t={} server_t={}".format(N, C, client_t, com_t, server_t))
        f.write(str(N)+', '+str(C)+', '+str(client_t)+', '+str(com_t)+', '+str(server_t)+'\n')
f.close()
