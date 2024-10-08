# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Mahdi 
"""

SERVER_IP = "127.0.0.1"
SERVER_PORT_FOR_CLIENT = 10000
SERVER_PORT_FOR_COMMITTEE = 10001
MAX_CONNECTIONS = 500
validation_file = "data/valid.p"
testing_file = "data/test.p"
training_files_location = "data/"
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
EPOCHS = 10
MAX_DB_SPLITS = 20

RAND_BIT_SIZE = 16
rand_low = -100.0
rand_high = +100.0

num_rounds = 1
num_participants = 40
committee_size = 20
leading_bits = 0
