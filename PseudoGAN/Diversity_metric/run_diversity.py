# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:23:47 2022

"""

import Diversity_lib_reimplemented as dl
import numpy as np
import os

isGenData=False
useImageNet=False
controlSize= 50
classidx=20
# gen_idx=0
bias=0.60
runs=5
fs_array=[]
ms_array=[]
diversity_array=[]
gamma_array=[]
for gen_idx in range(runs):
    #Load control data and generate similarity dictionary
    print ("Fetching and building data dict...")
    imageToFeatures,attributeLabels=dl.control_query_data(bias, isGenData , useImageNet, classidx, gen_idx, controlSize)
    
    
    #Generate similarity dictionaries
    print ("Generating sim dict...")
    simDict=dl.generateSimDict(imageToFeatures)
    mu_same,mu_diff,gamma=dl.generateMuSameDiff(attributeLabels,simDict)
    
    #Index control and generated position
    # testPos, controlPos = np.arange(controlSize,len(imageToFeatures),1), np.arange(0,controlSize,1)
    
    controlkeys=list(imageToFeatures.keys())[0:controlSize]
    testkeys=list(imageToFeatures.keys())[controlSize:]
    
    print ("Generating Controlled Bounds...")
    # def condition(g,s):
    #     return s > 0
    
    #Changed after first run of data collection (realised the direction was wrong)
    def condition(g,s):
        return s < 1
    
    
    lowerFS, lowerMS, upperFS, upperMS = dl.getBounds(controlkeys, condition, attributeLabels, simDict )
    fs,ms,Diversity=dl.audit_for_f(simDict, attributeLabels, testkeys, controlkeys, lowerFS, lowerMS, upperFS, upperMS, condition)
    True_D=bias-(1-bias)
    print ("Approximated Diversity= %f"%Diversity)
    print ("True Diversity= %f" %True_D)
    diversity_array.append(Diversity)
    fs_array.append(fs)
    ms_array.append(ms)
    gamma_array.append(gamma)

np.savez("./logs/class{}_bias{}_controlsize{}.npz".format(classidx,int(bias*10),controlSize),x=np.array(diversity_array))