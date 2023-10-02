# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:38:21 2022

"""

import os
import torch
import tqdm
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import torchvision
import torchvision.transforms as transforms

N_IMAGES = 30000
IMG_SIZE = 64
ATTR_PATH = 'attributes.pt'
#Real Data
data_dir="H:\Datasets\CelebA-HQ\data256x256"

def preprocess_images():
    # automatically save outputs to "data" directory
    IMG_PATH = os.path.join('./celebaHQ_{0}x{0}.pt'.format(IMG_SIZE))
    LABEL_PATH = os.path.join('./labels_celebaHQ_{0}x{0}.pt'.format(IMG_SIZE))

    
    print('preprocessing...')
    # NOTE: datasets have not yet been normalized to lie in [-1, +1]!
    transform = transforms.Compose(
        [
        # transforms.CenterCrop(140),
        transforms.Resize(IMG_SIZE)])

    data = np.zeros((N_IMAGES, 3, IMG_SIZE, IMG_SIZE), dtype='uint8')
    # labels = np.zeros((N_IMAGES, 40))
    
    labels=pd.read_csv("./CelebAMask-HQ-attribute-anno.csv").iloc[:,1:].to_numpy()
    
    print('starting conversion...')
    files=os.listdir(data_dir)
    for i in tqdm.tqdm(range(len(files))):
        with Image.open(os.path.join(data_dir,files[i])) as img:
            if transform is not None:
                img = transform(img)
        img = np.array(img)
        data[i] = img.transpose((2, 0, 1))
        # labels[i] = attr_data[i]
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    print("Saving images to {}".format(IMG_PATH))
    torch.save(data, IMG_PATH)
    torch.save(labels, LABEL_PATH)
    

preprocess_images()