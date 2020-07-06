from PIL import Image
import os
import os.path
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from collections import Counter


class CIFAR10Anom(CIFAR10):
    def __init__(self, root, stage='train',  transform=None, target_transform=None,
                 download=False, anom_classes=None, valid_split= 0.01,anom_ratio=0, seed=0):
        np.random.seed(seed)
        train = True if stage == 'train' or stage == 'valid' else False
        super(CIFAR10Anom, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                 download=download)
        
        if len(set(anom_classes) & set(self.class_to_idx.keys()))==0:
            print('No anomaly class found, will be trained on all the classes')
        
        anom_mapping= dict((i,-1) if c in anom_classes else (i,1) for c,i in self.class_to_idx.items())
        
        if train:
            self.targets = [anom_mapping[key] for key in self.targets]
            
            indices = list(range(len(self.data)))
            num_valid_indices = int(len(indices)*valid_split)
            valid_indices= np.random.choice(indices,num_valid_indices)
            train_indices = list(set(indices) - set(valid_indices))
#             print(len(train_indices),len(valid_indices))
        
        if stage == 'train':

            norm_indices = [i for i in train_indices if self.targets[i] == 1]
            anom_indices = [i for i in train_indices if self.targets[i] == -1]
            subsamples_anom_indices = np.random.choice(anom_indices, int(len(anom_indices)*anom_ratio))
#             print(len(norm_indices),len(anom_indices),len(subsamples_anom_indices))
            train_indices = norm_indices.copy() 
            train_indices.extend(subsamples_anom_indices)
            train_indices = np.array(train_indices)[torch.randperm(len(train_indices))]
#             print(len(train_indices))
            self.data = self.data[train_indices]
            self.targets = np.array(self.targets)[train_indices]

            
        elif stage == 'valid':
            
            self.data = self.data[valid_indices]
            self.targets = np.array(self.targets)[valid_indices]

        elif stage == 'test':
            self.targets = [anom_mapping[key] for key in self.targets]
        else:
            print(f'invalid stage: {stage}')
            raise