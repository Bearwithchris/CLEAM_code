# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:28:25 2022
"""

import numpy as np
from PIL import Image
import os
import tqdm

dataDir="../GeneratedData/StyleGANv2_celebaHQ"
files=os.listdir(os.path.join(dataDir,"samples"))
IMG_SIZE=128

data = np.zeros((len(files), IMG_SIZE, IMG_SIZE,3), dtype='uint8')

for i in tqdm.tqdm(range(len(files))):
    with Image.open(os.path.join(dataDir,"samples",files[i])) as img:
        img = np.array(img)
    # data[i] = img.transpose((2, 0, 1))
    # labels[i] = attr_data[i]
    data[i]=img

np.savez(os.path.join(dataDir,"np","generated_images"),x=np.moveaxis(data,3,1))

