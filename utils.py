# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Mahdi 
"""

import pickle
import socket
import struct

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    return msg

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)
    # print(msg[0], 'sent to', sock.getpeername())

# Determine the shape of an input list
def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else []
    return shape
    
def transpose_list(lst, axes=None):
    # Recursive function to build a nested list with the given shape
    def build_list(shape):
        if len(shape) == 1:
            return [None] * shape[0]
        return [build_list(shape[1:]) for _ in range(shape[0])]

    # Helper function to set a value in a nested list at a given index
    def set_value(lst, idx, value):
        for i in idx[:-1]:
            lst = lst[i]
        lst[idx[-1]] = value

    # Recursive function to get values from the original list
    def get_value(lst, idx):
        for i in idx:
            lst = lst[i]
        return lst

    # Recursive function to transpose the list
    def recursive_transpose(lst, current_idx, axes, shape):
        if len(current_idx) == len(shape):
            transposed_idx = [current_idx[ax] for ax in axes]
            value = get_value(lst, current_idx)
            set_value(transposed_lst, transposed_idx, value)
        else:
            for i in range(shape[len(current_idx)]):
                recursive_transpose(lst, current_idx + [i], axes, shape)

    # Validate input
    if not isinstance(lst, list):
        raise ValueError("Input must be a list.")

    # Get the shape of the list
    shape = get_shape(lst)

    # Set default axes if not provided
    if axes is None:
        axes = list(reversed(range(len(shape))))

    # Validate axes
    if len(axes) != len(shape):
        raise ValueError("Axes length must match list dimensions.")

    # Compute the shape of the transposed list
    transposed_shape = [shape[ax] for ax in axes]

    # Initialize the transposed list
    transposed_lst = build_list(transposed_shape)

    # Perform the transposition
    recursive_transpose(lst, [], axes, shape)

    return transposed_lst