# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Mahdi 
"""

import copy
import numpy as np
import torch
import random
from models import *

# def split_additive_secret(secret, n):
#     shares = [random.randint(0, secret) for _ in range(n - 1)]
#     shares.append(secret - sum(shares))
#     return shares

# def reconstruct_additive_secret(shares):
#     return sum(shares)

# def split_additive_mask(mask, n):
#     shares_dict = {}
#     for name, param in mask.named_parameters():
#         shares = [split_additive_secret(param.data[i].item(), n) for i in range(param.data.numel())]
#         shares_dict[name] = shares
#     return shares_dict

# def reconstruct_additive_mask(shares_dict):
#     reconstructed_model = BasicNet()  # Create a new instance of the model to hold the reconstructed parameters
#     for name, shares_list in shares_dict.items():
#         param_shape = shares_list[0].shape  # Get the shape of the parameter
#         reconstructed_param = torch.zeros(param_shape)  # Initialize the parameter with zeros
#         for i in range(param_shape[0]):
#             for j in range(param_shape[1]):
#                 shares = [shares_list[k][i][j] for k in range(len(shares_list))]
#                 reconstructed_param[i][j] = reconstruct_additive_secret(shares)  # Combine the shares to reconstruct the parameter
#         setattr(reconstructed_model, name, torch.nn.Parameter(reconstructed_param))
#     return reconstructed_model

# def pairwise_add_shares(shares_dicts):
#     combined_shares_dict = shares_dicts[0].copy()
#     for shares_dict in shares_dicts[1:]:
#         for name, shares_list in shares_dict.items():
#             combined_shares_list = combined_shares_dict[name]
#             for i in range(len(shares_list)):
#                 for j in range(len(shares_list[i])):
#                     combined_shares_list[i][j] += shares_list[i][j]
#     return combined_shares_dict        

# def flatten_masks(masks):
#     return [mask for masks_row in masks for mask in masks_row]

# def split_list_masks(masks, n):
#     flat_masks = flatten_masks(masks)
#     shares_list = [split_additive_secret(mask, n)]
#     return masks_splits

def split_additive_mask(mask, n, rand_low, rand_high):  # n*mask
    shares = [torch.randn(mask.shape) for _ in range(n - 1)]
    shares.append(mask - sum(shares))
    return shares

# def transpose_list(input_list):
#     return list(map(list, zip(*input_list)))

def are_tensors_equal(tensor1, tensor2):
    # Check if both are tensors and use torch.equal
    if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        return torch.allclose(tensor1, tensor2, rtol=1e-05, atol=1e-05)
    # If both are lists, check if they have the same length and recursively compare
    elif isinstance(tensor1, list) and isinstance(tensor2, list):
        if len(tensor1) != len(tensor2):
            return False
        return all(are_tensors_equal(t1, t2) for t1, t2 in zip(tensor1, tensor2))
    # If types do not match, they are not equal
    else:
        return False

def split_additive_masks(masks, n, rand_low, rand_high):    # n*P*mask
    all_shares = [split_additive_mask(mask, n, rand_low, rand_high) for mask in masks]
    return all_shares

from utils import get_shape

def pairwise_add_splitted_masks(splitted_masks):
    print("pairwise_add_splitted_masks:     get_shape(splitted_masks)={}".format(get_shape(splitted_masks)))
    sum_splitted_masks = splitted_masks[0]
    print("pairwise_add_splitted_masks:     get_shape(sum_splitted_masks)={}".format(get_shape(sum_splitted_masks)))
    for client_splitted_masks in splitted_masks[1:]:
        print("pairwise_add_splitted_masks:     get_shape(client_splitted_masks)={}".format(get_shape(client_splitted_masks)))
        sum_splitted_masks = pairwise_sum(sum_splitted_masks, client_splitted_masks)
        # for i in range(len(client_splitted_masks)):
        #     sum_splitted_masks[i] += client_splitted_masks[i]
    return sum_splitted_masks

def add_tensors(tensor1, tensor2):
    # Base case: if both elements are not lists, add them directly
    if not isinstance(tensor1, list) and not isinstance(tensor2, list):
        return tensor1 + tensor2

    # Recursive case: if both elements are lists, add their elements pairwise
    return [add_tensors(t1, t2) for t1, t2 in zip(tensor1, tensor2)]

def pairwise_sum(tensor_list1, tensor_list2):
    if len(tensor_list1) != len(tensor_list2):
        raise ValueError("The lists must have the same length.")
    
    return [add_tensors(t1, t2) for t1, t2 in zip(tensor_list1, tensor_list2)]

# def reconstruct_additive_mask(shares):
#     return sum(shares)

# def reconstruct_additive_masks(split_shares):
#     reconstructed_masks = []
#     for mask_shares in split_shares:
#         masks = []
#         for shares in zip(*mask_shares):  # Iterating through shares of each tensor
#             reconstructed_tensor = reconstruct_additive_mask(shares)
#             masks.append(reconstructed_tensor)
#         reconstructed_masks.append(masks)
#     return reconstructed_masks


def mask_model(seed, model, rand_low, rand_high):
    torch.manual_seed(seed)
    np.random.seed(seed)
    for parm in model.parameters():
        mask = torch.rand(parm.size()) * (rand_high - rand_low) + rand_low
        parm.data += mask

def unmask_model(seed, model, rand_low, rand_high):
    torch.manual_seed(seed)
    np.random.seed(seed)
    for parm in model.parameters():
        mask = torch.rand(parm.size()) * (rand_high - rand_low) + rand_low
        parm.data -= mask

def generate_masks_from_seed(seed, shapes, rand_low, rand_high):
    torch.manual_seed(seed)
    np.random.seed(seed)
    masks = [torch.rand(shape)*(rand_high-rand_low)+rand_low for shape in shapes]
    return masks

def sum_list_masks(list_masks):
    mask_size = len(list_masks[0])
    if len(list_masks) == 0:
        raise ValueError("The list of masks should not be empty.")
    masks_sum = [copy.deepcopy(list_masks[0][j]) for j in range(mask_size)]
    with torch.no_grad():
        for j in range(mask_size):
            masks_sum[j].zero_()
        for i in range(len(list_masks)):
            for j in range(mask_size):
                masks_sum[j] += list_masks[i][j]
    return masks_sum
