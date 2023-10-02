# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:15:02 2022
"""

import torch
import numpy as np
import os

#Select hyper parameters to work on
class_idx=20
IMG_SIZE=64

#Load data
path="./"
data = torch.load(path + 'celebaHQ_64x64.pt')
labels = torch.load(path + 'labels_celebaHQ_64x64.pt')
#Clean Labels before processing
labels = labels[:, class_idx]
_labels=np.zeros_like(labels)
for i in range(len(labels)):
    if labels[i]==1:
        _labels[i]=1
labels=torch.tensor(_labels)

#Filter out avaialble data to select from
label0=np.where(labels==0)[0]
label1=np.where(labels==1)[0]
samplesFromEachSplit=min(len(label0),len(label1))

split={"train":int(0),"test":int(0.8*samplesFromEachSplit),"val":int(0.9*samplesFromEachSplit)}
# split={"train":int(0),"test":0,"val":int(0.8*samplesFromEachSplit)}

# reproducibility
torch.manual_seed(777)
np.random.seed(777)


if os.path.isdir("./data/CelebAHQ/{}".format(class_idx))==0:
    try:
        os.mkdir("./data/")
    except:
        pass
    
    try:
        os.mkdir("./data/CelebAHQ/")
    except:
        pass    
    
    os.mkdir("./data/CelebAHQ/{}".format(class_idx))

pathPrefix="./data/CelebAHQ/{}/".format(class_idx)
#Train
_trainDataIdx=np.concatenate([label0[split['train']:split['test']],label1[split['train']:split['test']]])
torch.save(data[_trainDataIdx], pathPrefix+'{1}_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "train"))
torch.save(labels[_trainDataIdx], pathPrefix+'{1}_labels_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "train")) 
#Test
_testDataIdx=np.concatenate([label0[split['test']:split['val']],label1[split['test']:split['val']]])
torch.save(data[_testDataIdx], pathPrefix+'{1}_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "test"))
torch.save(labels[_testDataIdx], pathPrefix+'{1}_labels_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "test")) 
#val
_valDataIdx=np.concatenate([label0[split['val']:samplesFromEachSplit],label1[split['val']:samplesFromEachSplit]])
torch.save(data[_valDataIdx], pathPrefix+'{1}_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "val"))
torch.save(labels[_valDataIdx], pathPrefix+'{1}_labels_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE, "val")) 