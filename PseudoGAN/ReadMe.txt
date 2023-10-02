##PseudoGAN Experiment
#1) Setting Up the data
(a) Download the CelebA dataset here (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) into the `data/` directory (if elsewhere, note the path for step b). Of the download links provided, choose `Align&Cropped Images` and download `Img/img_align_celeba/` folder, `Anno/list_attr_celeba.txt`, and `Eval/list_eval_partition.txt` to `data/`.
(b) In path "./PseudoGAN/preprocess/": Preprocess the celebA data to output the data split. Use partitions {train,test,split}
```
python preprocess_celeba.py --data_dir=/path/to/downloaded/dataset/celeba/ --out_dir=../data --partition=train
```
(c)  In path "./PseudoGAN/preprocess/": Segment the data into it's respected SA i.e., classIdx. For example Gender=20 Blackhair=8 
```
python PreProcess_celeb_Multi.py --classIdx 20
```
(d) In path "./PseudoGAN/preprocess/": Create and sample from the PseudoGAN with p*=bias for SA=classIdx
```
python get_data_split.py --N 400 --S 30 --seed 777 --classIdx 20 --bias 0.9
```

#2) Prepare the classifier
(a) In path "./PseudoGAN/src/": Train the attribute classifier with ResNet-18 on SA=class_idx
```
python tran_attribute_clf.py --class_idx 20
```
(b) In path "./PseudoGAN/src/": Validate the accuracy (alpha) of the classifier with SA=class_idx. The accuracy with be ./PseudoGAN/logs/Attribute_classifier_accuracy.txt
```
python validate_acc.py --class_idx 20
```

#Run CLEAM
(a) In path "./PseudoGAN/src/": Pre-measure the phat values used in CLEAM with SA=class_ix and p*=bias
```
python measure_dist.py --class_idx=20 --bias 0.9 
```
(b) In path "./PseudoGAN/src/": Run CLEAM with the 1) accuracy measured earlier as the --acc input 2) The saved npzPath as per the example. This pythob script is a batchRun which does not require a specific bias to be stated
```
python CLEAM.py --acc [0.981437692125798,0.975407897848191] --npzPath ../data/relabelled/20/pred_dist/S30_N1000_Seed777/
```