# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:59:15 2022
https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb
"""

import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import dill as pickle

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    
BATCH_SIZE = 256
# # NUM_EPOCHS = 25
# LEARNING_RATE = 0.001
# NUM_WORKERS = 1


##########################
### Dataset
##########################

def norm_feat(features):
    return features/255.0


def build_multi_celeba_classification_datset(split):
    """
    Loads data for multi-attribute classification
    
    Args:
        split (str): one of [train, val, test] 
    
    Returns:
        TensorDataset for training attribute classifier
    """
    BASE_PATH = '../data/'
    data = torch.load(
        BASE_PATH + '{}_celeba_64x64.pt'.format(split))
    # print('returning labels for (black hair, gender) multi-attribute')
    labels = torch.load(
        BASE_PATH + '{}_multi_labels_celeba_64x64.pt'.format(split))
    dataset = torch.utils.data.TensorDataset(data, labels)
    return dataset

# def get_dataloaders_celeba(batch_size, num_workers=0,
#                            train_transforms=None,
#                            test_transforms=None,
#                            download=True):

#     if train_transforms is None:
#         train_transforms = transforms.ToTensor()

#     if test_transforms is None:
#         test_transforms = transforms.ToTensor()
    
#     def get_attr(attr):
#         return attr[20]

#     # get_smile = lambda attr: attr[20]

#     train_dataset = datasets.CelebA(root='.',
#                                     split='train',
#                                     transform=train_transforms,
#                                     target_type='attr',
#                                    # target_transform=get_attr,
#                                     download=download)

#     valid_dataset = datasets.CelebA(root='.',
#                                     split='valid',
#                                     target_type='attr',
#                                     #target_transform=get_attr,
#                                     transform=test_transforms)

#     test_dataset = datasets.CelebA(root='.',
#                                    split='test',
#                                    target_type='attr',
#                                  #  target_transform=get_attr,
#                                    transform=test_transforms)


#     train_loader = DataLoader(dataset=train_dataset,
#                               batch_size=batch_size,
#                               num_workers=num_workers,
#                               shuffle=True)

#     valid_loader = DataLoader(dataset=test_dataset,
#                              batch_size=batch_size,
#                              num_workers=num_workers,
#                              shuffle=False)
    
#     test_loader = DataLoader(dataset=test_dataset,
#                              batch_size=batch_size,
#                              num_workers=num_workers,
#                              shuffle=False)

#     return train_loader, valid_loader, test_loader


# train_loader, valid_loader, test_loader = get_dataloaders_celeba(
#     batch_size=BATCH_SIZE,
#     train_transforms=custom_transforms,
#     test_transforms=custom_transforms,
#     download=False,
#     num_workers=4)


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 1
learning_rate = 0.001
num_epochs = 3

# Architecture
num_features = 64*64
num_classes = 2

##########################
### MODEL
##########################


class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          # (1(32-1)- 32 + 3)/2 = 1
                          padding=1), 
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                          out_channels=64,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_2 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                          out_channels=128,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_3 = nn.Sequential(        
                nn.Conv2d(in_channels=128,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
          
        self.block_4 = nn.Sequential(   
                nn.Conv2d(in_channels=256,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),        
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))
        )
        
        self.block_5 = nn.Sequential(
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),            
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512,
                          out_channels=512,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=1),
                nn.ReLU(),   
                nn.MaxPool2d(kernel_size=(2, 2),
                             stride=(2, 2))             
        )
        
        
        self.classifierP1 = nn.Sequential(
                # nn.Linear(512*4*4, 4096),
                nn.Linear(512*2*2, 4096),
                nn.ReLU())
        
        self.classifierP21 = nn.Linear(4096, 4096)
        self.classifierP22 = nn.ReLU()
                
        self.classifierP3 = nn.Sequential(
                nn.Linear(4096, num_classes)
        )
            
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, np.sqrt(2. / n))
                m.weight.detach().normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.detach().normal_(0, 0.05)
                m.bias.detach().detach().zero_()
        
        
    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        # logits = self.classifier(x.view(-1, 512*4*4))
        x=self.classifierP1(x.view(-1, 512*2*2))
        x=self.classifierP21(x)
        feat=self.classifierP22(x)
        
        logits = self.classifierP3(feat)
        probas = F.softmax(logits, dim=1)

        return logits, probas

train_dataset = build_multi_celeba_classification_datset('train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = build_multi_celeba_classification_datset('test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


val_dataset = build_multi_celeba_classification_datset('val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
  
torch.manual_seed(random_seed)
model = VGG16(num_features=num_features,
              num_classes=num_classes)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

#Training

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = norm_feat(features).to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

start_time = time.time()
for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = norm_feat(features).to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        
        targets=targets.long()
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

        

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader),
              compute_accuracy(model, val_loader)))
        
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))


#Testing
with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))
    
    
torch.save(model.state_dict(), "./VGG16_CelebA_attractive.pt")