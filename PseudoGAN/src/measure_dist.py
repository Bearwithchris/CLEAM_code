# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 14:19:02 2022

"""

import time
import os
import glob
import sys

import functools
import math
import numpy as np
from tqdm import tqdm, trange
import pickle


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
import argparse

import metrics as fd


import sys
sys.path.append('../')
from clf_models import ResNet18, BasicBlock

def classify_examples(model, data):
    """
    classifies generated samples into appropriate classes 
    """
    model.eval()
    # data=np.moveaxis(data,1,-1)
    preds = []
    probs = []
    # if load_save==1:
    #     samples = np.load(sample_path)['x']
    # else:
    #     samples=X
    bs=100
    n_batches = data.shape[0] // bs
    remainder=data.shape[0]-(n_batches*bs)

    with torch.no_grad():
        # generate 10K samples
    
        for i in range(n_batches):
            x = data[i*bs:(i+1)*bs]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()

            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
        
        # print ("Pause")
           
        if remainder!=0:
            x = data[(i+1)*bs:(bs*(i+1))+remainder]
            samp = x / 255.  # renormalize to feed into classifier
            samp = torch.from_numpy(samp).to('cuda').float()
            # get classifier predictions
            logits, probas = model(samp)
            _, pred = torch.max(probas, 1) #Returns the max indices i.e. index
            probs.append(probas)
            preds.append(pred)
            
            
            
        preds = torch.cat(preds).data.cpu().numpy()
        probs = torch.cat(probs).data.cpu().numpy()
        # probs = torch.cat(probs).data.cpu()

    return preds, probs

def run():
    # Prepare state dict, which holds things like epoch # and itr #
    parser = argparse.ArgumentParser()
    parser.add_argument('--clf_path', type=str, help='Folder of the CLF file', default="attr_clf")
    parser.add_argument('--bias', type=float, help='To generate the individial dataset npz', default=0.6)
    parser.add_argument('--class_idx', type=int, default=20,
                        help='CelebA class label for training.')
    parser.add_argument('--out_dir', type=str, help='where to save outputs',default="./results/attr_clf")
    
    
    parser.add_argument('--N', type=int, help='batchsize.', default=1000)
    parser.add_argument('--S', type=int, help='number of batches.', default=30)
    parser.add_argument('--seed', type=int, help='random seed', default=777)
    args = parser.parse_args()

    # print ("test")
    CLF_PATH = os.path.join(args.out_dir,str(args.class_idx),"model_best.pth.tar")
    device = 'cuda'
    torch.backends.cudnn.benchmark = True


    #Log Runs
    # f=open('../../%s/log_stamford_fair.txt' %("logs"),"a")
    # fnorm=open('../../%s/log_stamford_fair_norm.txt' %("logs"),"a")
    # data_log=open('../../%s/log_stamford_fair_norm_raw.txt' %("logs"),"a")
    
    
    #Load classifier
    print('Pre-loading pre-trained single-attribute classifier...')
    clf_state_dict = torch.load(CLF_PATH)['state_dict']
    
    clf = ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False) 
    clf.load_state_dict(clf_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clf = clf.to(device)
    clf.eval()  # turn off batch norm
    
    #Load data batch
    DIR="../data/relabelled/"+str(args.class_idx)
    data=torch.load(os.path.join(DIR, 'gen_data_S%i_N%i_seed%i_%s_%f.pt'%(args.S,args.N,args.seed,args.class_idx,args.bias)))[0]
    
    distArray=[]
    for i in tqdm(range(len(data))):
        # train_set = torch.utils.data.TensorDataset(data[i])
        preds, probs = classify_examples(clf, data[i]) #Classify the data
        dist=fd.pred_2_dist(preds,2)
        distArray.append(dist)
        
    newDir=DIR+"/pred_dist/S%i_N%i_Seed%i"%(args.S,args.N,args.seed)
    if not os.path.isdir(newDir):
        os.makedirs(newDir)
    np.savez(newDir+"/pred_dist_S%i_N%i_seed%i_%s_%f.npz"%(args.S,args.N,args.seed,args.class_idx,args.bias),x=np.vstack(distArray))
    return np.vstack(distArray)
    

if __name__ == '__main__':
    out=run()