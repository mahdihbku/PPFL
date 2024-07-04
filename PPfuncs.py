# -*- coding: utf-8 -*-
"""
Created on May 8 2024

@author: Mahdi 
"""

import copy
import numpy as np
import torch
from models import *
from utils import get_shape

def split_additive_mask(mask, n, rand_low, rand_high):  # n*mask
    shares = [torch.randn(mask.shape) for _ in range(n - 1)]
    shares.append(mask - sum(shares))
    return shares

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
