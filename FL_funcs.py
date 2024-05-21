# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:40:56 2020

@author: abdullatif
"""

from tqdm import tqdm
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import *
import copy

device = torch.device("cpu")

def client_update(client_model, optimizer, train_loader, epoch=5):
    """
    This function is to train the local models using local data
    """
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()   

    
def server_aggregate(global_model, client_models):
    """
    This function aggregates the clients models and then form new global model
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)

def sum_models(models):
    if len(models) == 0:
        raise ValueError("At least one model is required.")
    model_sum = copy.deepcopy(models[0])
    with torch.no_grad():
        for param in model_sum.parameters():
            param.data.zero_()
        for model in models:
            for param_sum, param in zip(model_sum.parameters(), model.parameters()):
                param_sum.data += param.data 
    return model_sum

def replicate_model(original, copies):
    for model in copies:
        model.load_state_dict(original.state_dict())


def server_aggregate_masked(global_model, client_models, masks_sum):
    """
    This function aggregates the clients models and then form new global model
    """
    ### This will take simple mean of the weights of models ###
    model_sum = sum_models(client_models)
    with torch.no_grad():
        for (global_weight, sum_masked_weights, mask) in zip(
            global_model.parameters(), model_sum.parameters(), masks_sum
        ):
            global_weight.data = (sum_masked_weights.data - mask)/len(client_models)  # Element-wise subtraction
    
        
        
def test(global_model, test_loader):
    """This function evaluate the global model updated in the  server_aggregate()"""
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = global_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = correct / len(test_loader.dataset)

    return test_loss, acc    

def train_(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Train model
        model.train()
        losses, nums = zip(*[loss_batch(model, loss_func, x, y, opt) for x, y in train_dl])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # Validate the model during the training
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train loss: {train_loss:.6f}\t"
                  f"Validation loss: {valid_loss:.6f}\t",
                  f"Validation accruacy: {valid_accuracy:.3f}%")

def valid_batch(model, loss_func, x, y):
    output = model(x)
    loss = loss_func(output, y)
    pred = torch.argmax(output, dim=1)
    correct = pred == y.view(*pred.shape)   
    return loss.item(), torch.sum(correct).item(), len(x)

def to_device(x, y):
    return x.to(device), y.to(device, dtype=torch.int64)


def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(x)

def test_(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(*[valid_batch(model, loss_func, x, y) for x, y in dl])
        testing_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        testing_accuracy = np.sum(corrects) / np.sum(nums) * 100
        
    print(f"Testing loss: {testing_loss:.6f}\t"
          f"Testing accruacy: {testing_accuracy:.3f}%")
    return testing_accuracy,testing_loss