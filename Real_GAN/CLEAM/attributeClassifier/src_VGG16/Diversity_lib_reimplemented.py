# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:02:05 2022
"""

import numpy as np
import os, sys, glob, re
from scipy.spatial import distance

from tqdm.notebook import tqdm
from IPython.display import Markdown, display
from matplotlib.pyplot import figure
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model

import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure

import cv2

import torch
import time

import VGG_16_Model as modVGG
from torchvision.models import vgg16
from torchvision import transforms
import torch.nn.functional as F

attributesDict={20:"Gender",8:"Blackhair"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#To use PCA or not
pcaLogits=1

def norm_feat(features,imagenet):
    if imagenet==True:
        #Reference: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/35
        features=features/255.0
        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(features)
    else:
        return features/255.0

# def getModel():
# 	#Load the VGG model
# 	return VGG16(weights='imagenet', include_top=True)	

def getFeatureExtractor(class_idx,imagenet):
    if imagenet==True:
        model=vgg16(pretrained=True).to(device)
        model.classifier=model.classifier[:-1]
        
    else:
        PATH="./VGG16_CelebA_%s.pt"%attributesDict[class_idx]
        model = modVGG.recover_VGG(PATH)
    
    #Original implementation
    # model=VGG16(weights='imagenet', include_top=True)	
    
    model.eval()
    # feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
    # return feat_extractor
    return model

# def loadImage(path, target_size):
#  	img = image.load_img(path, target_size=target_size)
#  	x = image.img_to_array(img)
#  	x = np.expand_dims(x, axis=0)
#  	x = preprocess_input(x)
#  	return img, x

# def processImage(data,target_size):
#     data=np.transpose(data,(0,2,3,1))
#     data=preprocess_input(data)
#     return data

# def getFeature(inp):
#     image_path = inp[0]
#     feat_extractor = inp[1]
#     target_size = inp[2]
#     try:
#         img, x = loadImage(image_path, target_size);
#         feat = feat_extractor.predict(x)[0]

#         return feat
#     except Exception as e:
#         print (inp, e)
#         return []

'''
Generating control data from validation dataset
output: dictionary key=control_xx values= 300 vector of feature/ labels
'''
def control_query_data(bias, isGenData , imagenet, class_idx, gen_idx=0, controlSize=50, queriedSize=6000):
    m=int(controlSize/2)

    # PCA to reduce vector size
    pca = PCA(n_components=300)
    
    #Model intialiser (VGG-16 Feat extractor)
    feat_extractor = getFeatureExtractor(class_idx,imagenet)
    feat_extractor=feat_extractor.to(device)
    # model = getModel()
    # target_size=model.input_shape[1:3]
    
    imageToFeatures = {}
    attributeLabels= {}
    
    #Control data---------------------------------------------------------------------------------------
    BASE_PATH = '../data/'
    data = torch.load(BASE_PATH +"CelebAHQ/{}/".format(class_idx) + '{}_celebaHQ_64x64.pt'.format("val")).numpy()
    labels = torch.load(BASE_PATH +"CelebAHQ/{}/".format(class_idx) + '{}_labels_celebaHQ_64x64.pt'.format("val")).numpy()
    # labels = labels[:, class_idx]

    #Filter even balanced data
    zero_label=np.where(labels==0)[0]
    selected_zero_label=np.random.choice(zero_label,m)
    zero_data=data[selected_zero_label]
    
    #Build the new dataset
    one_label=np.where(labels==1)[0]
    selected_one_label=np.random.choice(one_label,m)
    one_data=data[selected_one_label]
    concat_data=np.vstack((zero_data,one_data))
    control_names=["Control_%i"%i for i in range(len(concat_data))]
    
    #Preprocess and transpose data
    # concat_control_data=processImage(concat_data,target_size)
    concat_control_data=concat_data
    concat_control_labels=np.concatenate((np.zeros_like(selected_zero_label),np.ones_like(selected_one_label)))
    #Control data---------------------------------------------------------------------------------------
    
    #Queried data--------------------------------------------------------------------------------------
    if isGenData==True:
        data=torch.load("../../Generated_Data/StyleGANv2_celebaHQ_to_publish_10k/np/generated_images_10k.pt")
        # selectedIndex=np.random.randint(len(data),size=queriedSize)
        selectedIndex=np.random.choice(np.arange(0,len(data)),size=queriedSize,replace=True)
        data=data[selectedIndex]
        data=F.interpolate(data, size=64).numpy()
        #Dummy labels
        labels=np.zeros(data.shape[0])
    else:
        data,labels = torch.load("../data/relabelled/20/gen_data_S30_N200_seed777_20_%f.pt"%bias)
        data,labels=np.vstack(list(data)),np.concatenate(list(labels))
    # data,labels=data.numpy(),labels.numpy()
    gen_names=["Gen_%i"%i for i in range(len(data))]
    
    #Preprocess and transpose data
    # queried_data=processImage(data,target_size)
    queried_data=data
    queried_labels=labels
    
    #Queried data---------------------------------------------------------------------------------------
    
    
    #Comibine datasets
    names=control_names+gen_names
    data=np.vstack((concat_control_data,queried_data))
    labels=np.concatenate((concat_control_labels,queried_labels))

    #debug
    probasList=[]
    for i in tqdm(range(len(names))):
        #Extract Feat vector (resize + expand dim + predict)
        # imageToFeatures[names[i]]=feat_extractor.predict(np.expand_dims(cv2.resize(data[i],target_size),axis=0))[0]
        # imageToFeatures[names[i]]=feat_extractor(np.expand_dims(cv2.resize(data[i],target_size),axis=0))[0]


        if imagenet==True:
            image=torch.unsqueeze(norm_feat(torch.tensor(data[i]),imagenet),dim=0).to(torch.float).to(device)
            imageToFeatures[names[i]] = feat_extractor(image)
        else:
            image=torch.tensor(np.expand_dims(norm_feat(data[i],imagenet),axis=0)).float().to(device)
            __,probas,imageToFeatures[names[i]] = feat_extractor(image) #logits, probas, feat
            probasList.append(probas.cpu().detach())
        imageToFeatures[names[i]]=imageToFeatures[names[i]].detach().cpu().numpy()[0]
        attributeLabels[names[i]]=labels[i]
    
    #PCA reduction
    imagePaths = list(imageToFeatures.keys())
    # np.argmax(np.vstack(np.array(probasList)),axis=1)
    features = list(imageToFeatures.values())
    if pcaLogits==1:
        features = pca.fit_transform(features)
    else:
        features =np.array(features)
    for i, path in enumerate(imagePaths):
        imageToFeatures[path] = features[i]
        
    return(imageToFeatures,attributeLabels)
    
# def control_data(class_idx=20,controlSize=50):
#     class_idx=20
#     m=int(500/2)
    
#     # PCA to reduce vector size
#     pca = PCA(n_components=300)
    
#     #Model intialiser (VGG-16 Feat extractor)
#     feat_extractor = getFeatureExtractor()
#     model = getModel()
#     target_size=model.input_shape[1:3]
    
#     imageToFeatures = {}
#     attributeLabels= {}
    
#     BASE_PATH = '../data/'
#     data = torch.load(BASE_PATH + '{}_celeba_64x64.pt'.format("val")).numpy()
#     labels = torch.load(BASE_PATH + '{}_labels_celeba_64x64.pt'.format("val")).numpy()
#     labels = labels[:, class_idx]
#     attributes=2
#     class_count=2
    
#     #Filter even balanced data
#     zero_label=np.where(labels==0)[0]
#     selected_zero_label=np.random.choice(zero_label,m)
#     zero_data=data[selected_zero_label]
    
#     #Build the new dataset
#     one_label=np.where(labels==1)[0]
#     selected_one_label=np.random.choice(one_label,m)
#     one_data=data[selected_one_label]
#     concat_labels=np.concatenate((np.zeros_like(selected_zero_label),np.ones_like(selected_one_label)))
#     concat_data=np.vstack((zero_data,one_data))
#     control_names=["Control_%i"%i for i in range(len(concat_labels))]
    
#     #Preprocess and transpose data
#     concat_data=processImage(concat_data,target_size)
    
#     for i in tqdm(range(len(control_names))):
#         #Extract Feat vector (resize + expand dim + predict)
#         imageToFeatures[control_names[i]]=feat_extractor.predict(np.expand_dims(cv2.resize(concat_data[i],target_size),axis=0))[0]
#         attributeLabels[control_names[i]]=concat_labels[i]
    
#     #PCA reduction
#     imagePaths = list(imageToFeatures.keys())
#     features = list(imageToFeatures.values())
#     features = pca.fit_transform(features)
#     for i, path in enumerate(imagePaths):
#         imageToFeatures[path] = features[i]
    
#     return(imageToFeatures,attributeLabels)

# def query_data(class_idx=20,gen_idx=0):
    
#     # PCA to reduce vector size
#     pca = PCA(n_components=300)
    
#     #Model intialiser (VGG-16 Feat extractor)
#     feat_extractor = getFeatureExtractor()
#     model = getModel()
#     target_size=model.input_shape[1:3]
    
#     imageToFeatures = {}
#     attributeLabels= {}
        
        
#     data,labels = torch.load('../data/resampled_ratio/gen_data_%i_%s'%(2,gen_idx))
#     data,labels=data.numpy(),labels.numpy()
#     gen_names=["Gen_%i"%i for i in range(len(data))]
    
#     #Preprocess and transpose data
#     data=processImage(data,target_size)
    
#     for i in tqdm(range(len(gen_names))):
#         #Extract Feat vector (resize + expand dim + predict)
#         imageToFeatures[gen_names[i]]=feat_extractor.predict(np.expand_dims(cv2.resize(data[i],target_size),axis=0))[0]
#         attributeLabels[gen_names[i]]=labels[i]
        
#     #PCA reduction
#     imagePaths = list(imageToFeatures.keys())
#     features = list(imageToFeatures.values())
#     features = pca.fit_transform(features)
#     for i, path in enumerate(imagePaths):
#         imageToFeatures[path] = features[i]

#     return(imageToFeatures,attributeLabels)

'''
Generate Similarity dictionary to lookup
Input: Lookup table,imgid1, imgid2
'''
def sim(imageToFeatures,img1, img2):
    return 2 - distance.cosine(imageToFeatures[img1], imageToFeatures[img2])

def generateSimDict(imageToFeatures):
    # print ("Generating similarity Dictionary...")
    simDict = {}
    imagePaths=list(imageToFeatures.keys())
    for img1 in tqdm(imagePaths):
        d = {}
        for img2 in imagePaths:
            d[img2] = sim(imageToFeatures,img1, img2)
        simDict[img1] = dict(d)
    return simDict

'''
Generate the Mu_same and Mu diff for a given label
Input: attribute dictionary (imgname=label), similary dictionry
'''
def generateMuSameDiff(attributeLabels,simDict):
    mu_same, mu_diff = [], []
    imagePaths=list(attributeLabels.keys())
    
    for img1 in tqdm(imagePaths):
        for img2 in imagePaths:
            if attributeLabels[img1] == attributeLabels[img2]:
                mu_same.append(simDict[img1][img2])
            else:
                mu_diff.append(simDict[img1][img2])
                
    print ("$\mu_{same}$", np.mean(mu_same), np.std(mu_same))
    print ("$\mu_{diff}$", np.mean(mu_diff), np.std(mu_diff))
    gamma = np.mean(mu_same) - np.mean(mu_diff)
    
    return mu_same,mu_diff,gamma



def getBounds(control, condition, attributeLabels, simDict ):
    # control=list(test.keys())
    validation_m, validation_f = [], []
    for i in range(len(control)):
        #Extract gender and skin labels from control
        g= attributeLabels[control[i]]
        
        if condition(g,g):
            #segment females/males 
            validation_f = validation_f + [control[i]]
        else:
            validation_m = validation_m + [control[i]]

    #Finding the similarity in all the controlled samples
    #1) Extract the similarity matrix of the validation samples (img1), given that the samples isn't comparing against itself
    #2) Iterrate throguh the second layer of comaprisons (img2)
    simMatrix = np.array([np.array([simDict[img1][img2] for img1 in validation_m if img1 != img2]) for img2 in control])   
    #Average similarity for all the samples
    simSum = [np.mean(simMatrix[i]) for i in range(len(simMatrix))]
    #Average score, if skin tone is greater than 3
    fs = np.mean([simSum[i] for i in range(len(simSum)) if condition(attributeLabels[control[i]], attributeLabels[control[i]])])
    #Average score if skin tone is less than 3
    ms = np.mean([simSum[i] for i in range(len(simSum)) if not condition(attributeLabels[control[i]], attributeLabels[control[i]])])
    #Average sim score male with skin:(>3/<3)
    lowerFS, upperMS = fs, ms
    lowerR = fs/(fs+ms)
    lowerD1 = min(fs/ms, ms/fs)

    #Repeat for female
    simMatrix = np.array([np.array([simDict[img1][img2] for img1 in validation_f if img1 != img2]) for img2 in control])
    simSum = [np.mean(simMatrix[i]) for i in range(len(simMatrix))]
    fs = np.mean([simSum[i] for i in range(len(simSum)) if condition(attributeLabels[control[i]], attributeLabels[control[i]])])
    ms = np.mean([simSum[i] for i in range(len(simSum)) if not condition(attributeLabels[control[i]], attributeLabels[control[i]])])
    #Average sim score Female with skin:(>3/<3)
    upperFS, lowerMS = fs, ms
    upperR = fs/(fs+ms)
    lowerD2 = min(fs/ms, ms/fs)
    
    return lowerFS, lowerMS, upperFS, upperMS

# Normalize \hat{d}(S) values
def calibrate(ys, lower, upper):
    return (ys - lower) / (upper - lower)

# DivScore for dataset with a given fraction of elements with z=0
# More positive means it favours fs (label=0), more negative means it favours ms (label=1)
def audit_for_f(simDict, attributeLabels, test, control, lowerFS, lowerMS, upperFS, upperMS, condition):
    # N = int(size*f)
    # random.shuffle(test)
    # dataset = []
    # dataset += [imagePaths[i] for i in test if condition(genderLabels[imagePaths[i]], skintoneLabels[imagePaths[i]])][:N]
    # dataset += [imagePaths[i] for i in test if not condition(genderLabels[imagePaths[i]], skintoneLabels[imagePaths[i]])][:(size-N)]
    # random.shuffle(dataset)
    dataset=test

    simMatrix = np.array([[simDict[img1][img2] for img1 in dataset] for img2 in control])
    simSum = np.mean(simMatrix, axis=1)
    fs = np.mean([simSum[i] for i in range(len(simSum)) if condition(attributeLabels[control[i]], attributeLabels[control[i]])])  #True if "female" i.e., label0
    ms = np.mean([simSum[i] for i in range(len(simSum)) if not condition(attributeLabels[control[i]], attributeLabels[control[i]])])

    fs = calibrate(fs, lowerFS, upperFS)
    ms = calibrate(ms, lowerMS, upperMS)
    return fs,ms,fs-ms
