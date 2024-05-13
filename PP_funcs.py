# -*- coding: utf-8 -*-
"""
Created on  May 8 2024

@author: Mahdi 
"""

import copy
import numpy as np
import torch

def mask_model(seed, model, rand_high, rand_low):
    torch.manual_seed(seed)
    np.random.seed(seed)
    for parm in model.parameters():
        mask = torch.rand(parm.size()) * (rand_high - rand_low) + rand_low
        parm.data += mask

def unmask_model(seed, model, rand_high, rand_low):
    torch.manual_seed(seed)
    np.random.seed(seed)
    for parm in model.parameters():
        mask = torch.rand(parm.size()) * (rand_high - rand_low) + rand_low
        parm.data -= mask

def generate_masks_from_seed(seed, shapes, high, low):
    torch.manual_seed(seed)
    np.random.seed(seed)
    masks = [torch.rand(shape)*(high-low)+low for shape in shapes]
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