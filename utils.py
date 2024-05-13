import pickle
import socket
import struct

SERVER_IP = "127.0.0.1"
SERVER_PORT = 10000
MAX_CONNECTIONS = 5
testing_file = "data/test.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"
training_files_location = "data/"
BATCH_SIZE = 32
LR=0.001
MOMENTUM=0.9
epochs = 10

num_participants = 2
num_rounds = 3

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    return msg

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())
