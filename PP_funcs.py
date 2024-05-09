# -*- coding: utf-8 -*-
"""
Created on  May 8 2024

@author: Mahdi 
"""

import copy
import numpy as np
import torch

def generate_masks_from_seed(seed, shapes, high, low):
    torch.manual_seed(seed)
    np.random.seed(seed)
    masks = []
    for shape in shapes:
        mask = torch.rand(shape) * (high - low) + low
        masks.append(mask)
    return masks

def sum_list_masks(list_masks):
    masks_sum = []
    if len(list_masks) == 0:
        raise ValueError("The list of masks should not be empty.")
    for j in range(len(list_masks[0])):
        masks_sum.append(copy.deepcopy(list_masks[0][j]))
        masks_sum[j].zero_()
    with torch.no_grad():
        for i in range(len(list_masks)):
            for j in range(len(list_masks[0])):
                masks_sum[j] += list_masks[i][j]
    return masks_sum
