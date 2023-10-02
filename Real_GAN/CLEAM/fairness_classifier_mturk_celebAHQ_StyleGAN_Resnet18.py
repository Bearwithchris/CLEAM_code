# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:16:59 2022
PGAN class: https://github.com/facebookresearch/pytorch_GAN_zoo/blob/main/models/progressive_gan.py
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from clf_models import ResNet18, BasicBlock
import os
import cv2
import numpy as np
from tqdm import tqdm
import numpy as np

import click

# from torchvision import transforms

import torch.nn.functional as F

def CLEAM(at0,at1,muP,sigmaP,S=30):
    pstary=(muP-at0)/(1-at0-at1)
    pstarx=1-(pstary)
    
    zalpha=1.96
    #Calculate approximated CI for pstar
    lower=(muP-zalpha*sigmaP/np.sqrt(S)-(1-at1))/(at0-(1-at1))
    upper=(muP+zalpha*sigmaP/np.sqrt(S)-(1-at1))/(at0-(1-at1))  
    return pstarx,pstary,lower,upper



    
#Validated on real Data (val data)
attributeDict={
                "Gender":[0.9471627486437613,0.9829294755877035],
                "Blackhair":[0.8693939393939394,0.8848484848484849]} 

SADict={"Gender":20,
        "BlackHair":8}
#StyleGAN2 10k
# GT={"Gender":0.637032,
     # "Blackhair":0.643841}

#StyleSwim
GT={"Gender":0.6621664381394368,
    "Blackhair": 0.593291846851598}


use_gpu = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@click.command()
@click.option("--attribute", default="Gender", type=click.Choice(['Gender','Blackhair','Attractive','Young']), help="Select from LA classifier")
@click.option("--datadir", type=str, help="xxx.pt path pointing to generated images", default="../GeneratedData/StyleGANv2_celebaHQ/np/generated_images.npz")
@click.option("--classdir", type=str, help="path pointing to classifier directory", default="../CLEAM/attributeClassifier/src_ResNet-18/results/attr_clf/")
@click.option("--figheader", type=str)
@click.option("--seed", type=int, default=999)
def measure_dist(datadir,attribute,figheader,seed,classdir):
    
    #Select attribure classifier's accuracy
    accAt=attributeDict[attribute]
    
    #Load data
    # data=torch.load(datadir)
    data=np.load(datadir)['x']
    data=torch.tensor(data).to(torch.float)
    

    
    #Scale down data
    data=F.interpolate(data, size=64)
    
    #Load Classifier
    classifier=ResNet18(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2, grayscale=False)
    Cweights=torch.load(os.path.join(classdir,"{}".format(SADict[attribute]),"model_best.pth.tar"))['state_dict']
    classifier.load_state_dict(Cweights)
    classifier.to(device)
    
    #Label the data with Resnet-18 classifier
    index=0
    batchSize=100
    numBatches=int(len(data)/batchSize)
    hardLabelsArray=[]
    for i in tqdm(range(numBatches)):
        # print ("Classifying batch:{}...".format(i))
        _data=(data[i*batchSize:(i+1)*batchSize,:,:].float()/255).to(device)
        # _data=[i*batchSize:(i+1)*batchSize,:,:]
        logits,probas=classifier(_data)
        hard_Labels=np.argmax(probas.cpu().detach(),axis=1)
        index=saveImg(_data,hard_Labels,index)
        hardLabelsArray.append(np.argmax(probas.cpu().detach(),axis=1))
    
    #Hard lavel array
    hardLabelsArray=np.concatenate(np.array(hardLabelsArray))
    
    #Save hard labels
    np.savez("ClassifierLabel.npz",x=hardLabelsArray.tolist())
    
    #Generate the distribution for s=30
    batchSize=1000
    numBatches=30
    # numBatches=int(len(data)/batchSize)
    distArray=[]
    
    np.random.seed(seed)
    for i in tqdm(range(numBatches)):
        try:
            selectedArg=np.random.choice(len(hardLabelsArray),batchSize,replace=True)
        except:
            print ("Error")
        _hardLabelsArray=hardLabelsArray[selectedArg]
        label0=np.where(_hardLabelsArray==0)[0]
        label1=np.where(_hardLabelsArray==1)[0]
        dist=np.array([len(label0)/len(_hardLabelsArray),len(label1)/len(_hardLabelsArray)])
        distArray.append(dist)
    
    #Plot results 
    basePE,CLEAMPe,baseIE,CLEAMIE=plotDist(distArray,attribute,figheader)
    
    return basePE,CLEAMPe,baseIE,CLEAMIE

def saveImg(_data,labels,index):
    try:
        os.makedirs("labelled",)
        os.makedirs("./labelled/0")
        os.makedirs("./labelled/1")
    except:
        if index==0:
            print ("Directory labelled already exist....")
        else:
            pass
    
    for i in range(len(labels)):
        if labels[i]==1:
            plt.imsave("./labelled/1/%i.jpeg"%index,_data[i].detach().cpu().permute(1,2,0).numpy())
        else:
            plt.imsave("./labelled/0/%i.jpeg"%index,_data[i].detach().cpu().permute(1,2,0).numpy())
        index+=1
    return index
        

def plotDist(dist,attribute,figheader):
    
    #Select attribure classifier's accuracy
    accAt=attributeDict[attribute]
    
    phat0=np.array(dist)[:,0]
    
    #Baseline Statistics
    baselineMean=np.mean(phat0)
    baselineStd=np.std(phat0)
    
        
    
    plt.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(111)  
    CLEAMPoint,__,CLEAMLower,CLEAMUpper=CLEAM(accAt[0],accAt[1],baselineMean,baselineStd)
    ax1.hist(phat0,label=r"$\hat{p}_0$ ",color="r",alpha=0.3)
    
    
    
    
    #Plot CLEAM lines
    ax1.axvline(CLEAMLower, color='g', linestyle='solid', linewidth=2,label="CLEAM Interval Estimate")
    ax1.axvline(CLEAMUpper, color='g', linestyle='solid', linewidth=2)
    ax1.axvline(baselineMean-1.96*(baselineStd/np.sqrt(len(phat0))), color='b', linestyle='solid', linewidth=2,label="Baseline Interval Estimate")
    ax1.axvline(baselineMean+1.96*(baselineStd/np.sqrt(len(phat0))), color='b', linestyle='solid', linewidth=2)
    ax1.axvline(GT[attribute],color='black',linestyle="dotted",linewidth="4", label="GroundTruth(hand labelled)")
    
    ax1.set_ylabel("Count")
    ax1.set_xlabel(r"$\hat{p}_0$") 
    ax1.legend( loc=1,fontsize=30 )
    

    
    plt.savefig("./{}_BASELINE_CLEAM_dist_plot.pdf".format(figheader), bbox_inches='tight')
    plt.savefig("./{}_BASELINE_CLEAM_dist_plot.png".format(figheader), bbox_inches='tight')
    print ("Point Estimte------------------------------------")
    baselinePEE=abs((GT[attribute]-baselineMean)/GT[attribute])
    print ("Baseline: {:3f} Norm Err: {:2f}%".format(baselineMean,baselinePEE*100))
    CLEAMPEE=abs((GT[attribute]-((CLEAMLower+CLEAMUpper)/2))/GT[attribute])
    print ("CLEAM: {:3f}, Norm Err: {:2f}%".format((CLEAMLower+CLEAMUpper)/2,CLEAMPEE*100))
    print ("Interval Estimte----------------------------------")
    maxBaselineIEE=max(abs(GT[attribute]-((baselineMean-1.96*(baselineStd/np.sqrt(len(phat0))))))/GT[attribute],abs(GT[attribute]-((baselineMean+1.96*(baselineStd/np.sqrt(len(phat0))))))/GT[attribute])
    print ("Baseline range: [ {:3f} , {:3f} ], Max Norm Err: {:2f}%".format(baselineMean-1.96*(baselineStd/np.sqrt(len(phat0))),(baselineMean+1.96*(baselineStd/np.sqrt(len(phat0)))),maxBaselineIEE*100))
    maxCLEAMIEE=max(abs((GT[attribute]-CLEAMLower)/GT[attribute]),abs((GT[attribute]-CLEAMUpper)/GT[attribute]))
    print ("CLEAM range: [ {:3f} , {:3f} ], Max Norm Err: {:2f}%".format(CLEAMLower,CLEAMUpper,maxCLEAMIEE*100))
    return baselineMean,(CLEAMLower+CLEAMUpper)/2,[baselineMean-1.96*(baselineStd/np.sqrt(len(phat0))),(baselineMean+1.96*(baselineStd/np.sqrt(len(phat0))))],[CLEAMLower,CLEAMUpper]


if __name__=="__main__":

    #Generate the raw(BASELINE) LA distribution
    basePE,CLEAMPe,baseIE,CLEAMIE=measure_dist()
    

