# -*- coding: utf-8 -*-
"""
Created on  Dec 2

@author: abdullatif 
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 00:13:03 2020

@author: abdullatif
"""

import pickle 
from torch.utils.data.dataset import Dataset

class ProcessDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform
        
    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])

    def __len__(self):
        return self.count

def to_device(x, y):
    return x.to(device), y.to(device, dtype=torch.int64)


class PrepareDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))            