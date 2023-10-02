#GANs
##1)Training attribute classifier
a) Preparing the dataset
> Download the dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
> Pepare a pre-processed version with the following source code.
> In Path ./Real_GAN/CLEAM/attributeClassifier: Run the following with *data_dir* changed to the location of the images and *label_dir* as the label directory
```
python pre-process.py
```
> In Path ./Real_GAN/CLEAM/attributeClassifier: Split data into even train/test/val. Ammend class_idx to the respective SA.
```
python data_split.py
```
b) Train Attribute Classifier
>In Path ./Real_GAN/CLEAM/attributeClassifier/src_ResNet-18: train attribute classifier, change class_idx to the respective SA
```
python train_attribute_clf --class_idx 20
```
>In Path ./Real_GAN/CLEAM/attributeClassifier/src_ResNet-18: Validate attribute classifier, change class_idx to the respective SA
```
python validate_acc.py --class_idx 20
```

##2) Preparing data for real GAN
> Download dataset from annonymous link https://drive.google.com/drive/folders/1ENslNLyK6EEG2qj5YLZ3Qu3rFijJWEqB?usp=sharing and copy them into ./Real_GAN/GeneratedData/*datasetName*/samples
a)In Path ./Real_GAN/preprocess: Preprocess dataset to .npz format, edit dataDir for new dataset
```
python numpyData.py
```


##3) Run CLEAM
a) In Path ./Real_GAN/CLEAM
> Run for CLEAM approximation. Please edit attributeDict dictionary with the validated classifier's accuracy
```
python fairness_classifier_mturk_celebAHQ_StyleGAN_Resnet18.py
```



#DGM
#1) Download CelebA-HQ dataset
> Download dataset and 'CelebAMask-HQ-attribute-anno' from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

#2) Evaluate CLIP's accuracy
a)In path ./Real_GAN/CLEAM/CLIP
>Change the labelpath and dataPath in the python script to your paths
```
python celebAHQ_realsamples_labeller_measure_alpha.py
```

#3) Evluate the DGM generated samples 
a)In path ./Real_GAN/CLEAM/CLIP
> update the --acc to the measured accuracy in (2)
> update the --dataPath to where the data is located 
> update the --SA to the respective senstive attribute {Gender,Smiling}
```
python CLEAM_CLIP.py
```