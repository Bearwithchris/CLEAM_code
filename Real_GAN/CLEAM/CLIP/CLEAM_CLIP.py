# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#Clip imports
import clip
import torch
from PIL import Image

import argparse

def CLEAM(accAt,phat,N,S):
    
    def reverse(at1,at2,mu):
        pstary=(mu-at1)/(1-at1-at2)
        pstarx=1-(pstary)
        return (pstarx,pstary)
    
    # N=1000
    # files=os.listdir(prefix)
    
    #Filer Phat-0 data
    phat0=phat[:,0]
    # bias=float(files[0].strip(".npz").split("_")[-1])
    
    #Calculate statistics
    mup=np.mean(phat0)
    sigmap=np.std(phat0)
    
    #Calculate CLEAM
    pStar0,pStar1=reverse(accAt[0],accAt[1],mup)      
    zalpha=1.96
    #Calculate approximated CI for pstar
    lower=(mup-zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))
    upper=(mup+zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))      

    
    return (mup,sigmap,pStar0,lower,upper)   

def evaluate(accAt,phat,N,S):
    
    def fl2(p0):
        return np.sqrt((p0-0.5)**2+((1-p0)-0.5)**2)
    
    mup,sigmap,MLE0,low,high=CLEAM(accAt,phat,N,S)
    baselineLow=mup-1.96*(sigmap/np.sqrt(S))
    baselineHigh=mup+1.96*(sigmap/np.sqrt(S))
    
    print ("Baseline--------------------------------------------")
    print ("Baseline PE of  p*_0= {:.4f} -----> f={:.4f} ".format(mup,fl2(mup)))   
    print ("Baseline IE of p*_0= [ {:.4f} , {:.4f} ] -----> f= [ {:.4f} , {:.4f} ]".format(baselineLow,baselineHigh,fl2(baselineLow),fl2(baselineHigh)))
    
    print ("CLEAM--------------------------------------------")
    print ("CLEAM PE of  p*_0= {:.4f} -----> f={:.4f} ".format(MLE0,fl2(MLE0)))   
    print ("CLEAM IE of p*_0= [ {:.4f} , {:.4f} ] -----> f= [ {:.4f} , {:.4f} ]".format(low,high,fl2(low),fl2(high)))



SAClasses={"Gender":["a photo of a Female","a photo of a Male"],
           "Smiling": ["a photo of a face not smiling", "a photo of a face smiling"]
           }

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, help='dataPath', default="../../stable-diffusion-main/output/person/samples")
    parser.add_argument('--SA', type=str, choices=["Gender","Smiling"] , default="Smiling")
    parser.add_argument('--acc' , nargs="+", default=[0.9981911911006602,0.9798317807723614], type=float)
    parser.add_argument('--n', type=int, default="400")
    parser.add_argument('--s', type=int, default="30")
    # parser.add_argument('--bias' , default=0.9, type=float)
    args = parser.parse_args()
    
    #Selected SA
    selectedSA=args.SA
    
    #Load data
    dataPath=args.dataPath
    
    
    #Load Clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    #LabelPrompt
    text = clip.tokenize(SAClasses[selectedSA]).to(device)
    
    
    dataList=np.array(os.listdir(dataPath))
    # SAdist=np.zeros((1,2))

    classifierLabel=np.zeros(len(dataList))
    
    print ("Classifying all sampels...")
    for i in tqdm(range(len(dataList))):
        with torch.no_grad():
            # if int(dataList[i].strip('.jpg')) in images:
            _dataPath=os.path.join(dataPath, dataList[i])
            image = preprocess(Image.open(_dataPath)).unsqueeze(0).to(device)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            classifierHardLabel=np.argmax(probs)

            classifierLabel[i]=classifierHardLabel
            # tqdm.write(SAdist)
            # else:
                # print ("pass")
     
    phat=[]
    for i in range(args.s):      
        print ("Randomly sampling %i sampels..."%args.n)
        selectedIdx=np.random.choice(len(classifierLabel),args.n)
        _labels=classifierLabel[selectedIdx]
        SAdist=np.array([len(np.where(_labels==0)[0]),len(np.where(_labels==1)[0])])
        SAdist=SAdist/len(selectedIdx)
        phat.append(SAdist)
    
    phat=np.array(phat)
    # print ("Confusion Matrix for Classification of CelebA-HQ %i Samples (%s): "%(len(dataList),selectedSA)+str(SAdist))
    
    evaluate(args.acc,phat,args.n,args.s)
