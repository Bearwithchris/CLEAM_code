# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

#Clip imports
import clip
import torch
from PIL import Image

#Sensitve attribute dictionary
SA={"Gender":21
    }

SAClasses={"Gender":["a photo of a Female","a photo of a Male"],
           }


#Selected SA
selectedSA="Gender"


#Load data
labelpath="../../StyleGANv2/Prepare_data/CelebA-HQ/celebA-HQ/CelebAMask-HQ"
dataPath="../../StyleGANv2/Prepare_data/CelebA-HQ/celebA-HQ/data256x256"
labelsDf=pd.read_csv(os.path.join(labelpath, "CelebAMask-HQ-attribute-anno.csv"))
keys=list(labelsDf.keys())
selectedKey=keys[SA[selectedSA]]

#Load Clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#LabelPrompt
text = clip.tokenize(SAClasses[selectedSA]).to(device)

images=list(labelsDf[keys[0]])
filteredLabels=list(labelsDf[selectedKey])

#Filter the appropraite balance labels idx (note that there is a new order)
label0=np.where(np.array(filteredLabels)==0)[0]
label1=np.where(np.array(filteredLabels)==1)[0]
maxLen=min(len(label0),len(label1))
label0=label0[0:maxLen]
label1=label1[0:maxLen]
labelIdx=np.concatenate((label0,label1),axis=0) #Concat

#Filter images and labels according to index
images=np.array(images)[labelIdx]
images=[int(i.strip('.jpg')) for i in images]
filteredLabels=np.array(filteredLabels)[labelIdx]

# dataList=np.array(os.listdir(dataPath))
confusionMatrix=np.zeros((2,2))

for i in tqdm(range(len(filteredLabels))):
    with torch.no_grad():
        # if int(dataList[i].strip('.jpg')) in images:
        _dataPath=os.path.join(dataPath, f"{images[i]+1:05}.jpg")
        image = preprocess(Image.open(_dataPath)).unsqueeze(0).to(device)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        classifierHardLabel=np.argmax(probs)
        GTHardLabel=filteredLabels[i]
        # if GTHardLabel==0:
        #     print ("DEBUG")
        confusionMatrix[classifierHardLabel][GTHardLabel]+=1
        tqdm.write(confusionMatrix)
        # else:
            # print ("pass")
confusionMatrix=confusionMatrix/len(images)

print ("Confusion Matrix for Classification of CelebA-HQ %i Samples (%s): "%(len(images),selectedSA)+str(confusionMatrix))
