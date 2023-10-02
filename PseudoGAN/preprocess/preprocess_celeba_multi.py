# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:40:59 2021

"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from IPython.display import Image
import torch
import argparse




def genNewDataset(split_type):
    #Saving logs for training reference
    f=open(LOG_DIR+"Preprocessed_data_config.txt",'a')
    f.write("***********NEW ENTRY******************\n")
    f.write("Entry type: "+split_type)
    
    #Sanity check intended attributes
    attr_lookup=np.array(["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"
    ])
    attr_name=[]
    for attr in args.classIdx: attr_name.append(attr_lookup[attr])
    print("Selected attributes: " + str(attr_name))
    f.write(str(attr_name)+'\n')
    
    
    # Load data: make sure to have pre-processed the celebA dataset before running this code!
    data = torch.load(os.path.join(DATA_DIR, '{}_celeba_64x64.pt'.format(split_type)))
    labels = torch.load(os.path.join(DATA_DIR, '{}_labels_celeba_64x64.pt'.format(split_type)))
    new_labels = np.zeros(len(labels))
    unique_items = np.unique(labels[:,args.classIdx], axis=0)
    
    #Re-labelling the respective permutation with index [0,1,...]
    minCount=162770
    for i, unique in enumerate(unique_items):
        yes = np.ravel([np.array_equal(x,unique) for x in labels[:,args.classIdx]])
        new_labels[yes] = i
        
        count=len(yes[yes==True])
        
        #Determine the least amount of samples available per attribute
        if count<minCount:
            minCount=count
        print(unique, i, "count=%i"%count)
        f.write(str(unique)+" "+ str(i)+" count=%i \n"%count)
        
    #Create an even training data set
    even_args = np.zeros(minCount*len((np.unique(new_labels))))
    for i in range(len((np.unique(new_labels)))):
       even_args[minCount*i:minCount*(i+1)]=np.random.choice(np.where(new_labels==i)[0],minCount,replace=False)
    
    even_data= data[even_args,:,:,:] #Even data
    new_labels = torch.from_numpy(new_labels)
    even_labels=new_labels[even_args]
    f.write("Even Labels of TOTAL SAMPLES =%i \n"%len(even_labels))
    f.close()
    
    torch.save(new_labels, os.path.join(OUT_DIR, '{}_relabelled_new_labels_celeba_64x64.pt'.format(split_type)))
    torch.save(even_labels, os.path.join(OUT_DIR,'{}_relabelled_even_labels_celeba_64x64.pt'.format(split_type)))
    torch.save(even_data, os.path.join(OUT_DIR, '{}_relabelled_even_data_celeba_64x64.pt'.format(split_type)))

if __name__=="__main__":
    DATA_DIR = '../data/'
    OUT_DIR="../data/relabelled/"
    LOG_DIR = '../logs/'
    splits=["train","test","val"]
    
    #Prameter Parsers
    parser = argparse.ArgumentParser()
    parser.add_argument('--classIdx',nargs="*", type=int, help='CelebA class label for training.', default=[20])
    args = parser.parse_args()
    
    OUT_DIR=OUT_DIR+""+"_".join(str(e) for e in args.classIdx)
    
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    
    for i in splits:
        genNewDataset(i)
    