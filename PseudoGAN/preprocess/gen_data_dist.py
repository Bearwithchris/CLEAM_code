# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:27:51 2021

"""
import torch 
import numpy as np
import os
import argparse
import copy
from tqdm import tqdm

BASE_PATH = '../data/'
parser = argparse.ArgumentParser()
#Basic Hyper parameters
parser.add_argument('--N', type=int, help='batchsize.', default=1000)
parser.add_argument('--S', type=int, help='number of batches.', default=30)
parser.add_argument('--seed', type=int, help='random seed', default=777)

# parser.add_argument('--class_idx', type=int, help='CelebA class label for training.', default=20)
# parser.add_argument('--multi_class_idx',nargs="*", type=int, help='CelebA class label for training.', default=[8])

parser.add_argument('--classIdx',nargs="*", type=int, help='CelebA class label for training.', default=[20])
# parser.add_argument('--multi', type=int, default=1, help='If True, runs multi-attribute classifier')
parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
# parser.add_argument('--mode_constant', type=int, default=1, help='If True, normal mode, else extreme mode')
# parser.add_argument('--ABEP', type=int, default=0, help='State the Absolute bias starting point index')
# parser.add_argument('--step_mul', type=int, default=1, help='defines the dist step size')

#For bias one experiment
parser.add_argument('--bias', type=float, default=0.6, help='defines the size of the bias one dist')
args = parser.parse_args()


# def sample_max(dist):
#     class_idx=args.class_idx
#     split=args.split_type
#     if args.multi==0:
#         data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
#         labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
#         labels = labels[:, class_idx]
#         attributes=2
#         class_count=2
#     else:
#         data = torch.load(BASE_PATH + '{}_multi_even_data_celeba_64x64.pt'.format(split))
#         labels = torch.load(BASE_PATH + '{}_multi_even_labels_celeba_64x64.pt'.format(split))
#         attributes=2**(len(args.multi_class_idx))
#         class_count=len(args.multi_class_idx)
        
#     #Determine the number of samples per class (even)
#     minCount=162770
#     for i in range((attributes)):
#         count=len(np.where(labels==i)[0])
#         if count<minCount:
#             minCount=count
#     # cap=minCount/max(dist)
#     cap=minCount
#     return cap


def generate_test_datasets(new_data,new_labels,index,cap,rep):
    """
    Returns a dataset used for classification for given class label <class_idx>. If class_idx2 is not None, returns both labels (this is typically used for downstream tasks)
    
    Args:
        split (str): one of [train, val, test]
        class_idx (int): class label for protected attribute
        class_idx2 (None, optional): additional class for downstream tasks
    
    Returns:
        TensorDataset for training attribute classifier
    """
    #Retrieve database
    class_idx=args.classIdx
    split=args.split_type
    # if not args.multi:
    # #     data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format(split))
    # #     labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format(split))
    # #     labels = labels[:, class_idx]
    #     attributes=2
    #     class_count=2
    # else:
    # #     data = torch.load(BASE_PATH + '{}_multi_even_data_celeba_64x64.pt'.format(split))
    # #     labels = torch.load(BASE_PATH + '{}_multi_even_labels_celeba_64x64.pt'.format(split))
    #     attributes=2**(len(args.multi_class_idx))
    #     class_count=len(args.multi_class_idx)
    
    #Randomly selected data from the p* distribution
    label_arg=np.random.choice(np.arange(len(new_labels)),cap)
    selected_data= new_data[label_arg,:,:,:] #Even data
    selected_labels=new_labels[label_arg]

    # torch.save((selected_data,selected_labels),'../data/resampled_ratio/gen_data_%i_%s'%(attributes,index))
    
    # return new_labels
    pstar1ratio=float(selected_labels.sum()/len(selected_labels))
    gt_ratio=np.array([1-pstar1ratio,pstar1ratio])
    return selected_data,selected_labels,gt_ratio


#Multu===================================================================================

if __name__=='__main__':
    #Initialise parameters
    N=args.N
    S=args.S
    replace=True
    pstar=[args.bias,1-args.bias]
    np.random.seed(args.seed)

    #Load data (raw)
    class_idx=args.classIdx
    split=args.split_type

    #Load data
    DIR="../data/relabelled/"+""+"_".join(str(e) for e in args.classIdx)+"/"
    data = torch.load(DIR + '{}_relabelled_even_data_celeba_64x64.pt'.format(split))
    labels = torch.load(DIR + '{}_relabelled_even_labels_celeba_64x64.pt'.format(split))
    attributes=2**(len(args.classIdx))
    
    #Redundant loop, since even 
    count=[]
    for i in range((attributes)):
        count.append(len(np.where(labels==i)[0]))
    count=min(count)
        
    #Assigned the count of "population dataset"
    pstarCount=np.zeros(2)
    pstarCount[0]=count*pstar[0]
    pstarCount[1]=count*pstar[1]
    pstarCount=pstarCount.round()
    pstarCount=pstarCount.astype(int)
    
    #Select Arg of labels to sample out to make dataset with p* dist
    label_arg=np.ones(pstarCount.sum())
    point=0
    for i in range(attributes):
        label_arg[point:point+pstarCount[i]]=np.random.choice(np.where(labels==i)[0],pstarCount[i],replace=False)
        point=point+pstarCount[i]
    

    new_data= data[label_arg,:,:,:] #Even data
    new_labels=labels[label_arg]
    del(data)
    del(labels)

    
    gt_ratio_array=np.zeros((S,2))
    gt_dataset=np.zeros((S,N,3,new_data.shape[2],new_data.shape[3]))
    gt_labels=np.zeros((S,N))
    for i in tqdm(range(S)):
        selected_data,selected_labels,gt_ratio=generate_test_datasets(new_data,new_labels,i,N,replace)
        #Populate dist
        gt_ratio_array[i]=gt_ratio
        gt_labels[i]=selected_labels
        gt_dataset[i]=selected_data
        
    outpath=os.path.join(DIR, 'gen_data_S%i_N%i_seed%i_%s_%f.pt'%(S,N,args.seed,"_".join(str(e) for e in args.classIdx),args.bias))
    torch.save((gt_dataset,gt_labels),outpath)
    


