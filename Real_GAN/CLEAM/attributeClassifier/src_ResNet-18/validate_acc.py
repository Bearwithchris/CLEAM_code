import os
import sys
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from clf_models import BasicBlock, build_model
from utils import save_checkpoint
from dataset_splits import (
    build_celeba_classification_dataset,
    build_even_celeba_classification_dataset,
)

import matplotlib.pyplot as plt
import shutil

def saveImg(_data,labels,index):
    try:
        os.makedirs("labelled",)
        os.makedirs("./labelled/0")
        os.makedirs("./labelled/1")
    except:
        if index==0:
            print ("Directory labelled already exist....")
            shutil.rmtree("labelled")
            # shutil.rmtree("./labelled/0")
            # shutil.rmtree("./labelled/1")
            os.makedirs("labelled",)
            os.makedirs("./labelled/0")
            os.makedirs("./labelled/1")
        else:
            pass
    
    for i in range(len(labels)):
        if labels[i]==1:
            plt.imsave("./labelled/1/%i.jpeg"%index,_data[i].detach().cpu().permute(1,2,0).numpy())
        else:
            plt.imsave("./labelled/0/%i.jpeg"%index,_data[i].detach().cpu().permute(1,2,0).numpy())
        index+=1
    return index
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='celeba',default='celeba',)
    parser.add_argument('--out_dir', type=str, help='where to save outputs',default="./results/attr_clf")
    parser.add_argument('--ckpt-path', type=str, default='./results/multi_clf', 
                        help='if test=True, path to clf checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='minibatch size [default: 64]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs [default: 10]')
    parser.add_argument('--class_idx', type=int, default=20,
                        help='CelebA class label for training.')
    parser.add_argument('--even', type=int, default=1, 
                        help='If True, runs multi-even-attribute classifier')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--split_type', type=str, help='[train,val,split]', default="test")
    
    #Simulate
    # argv = ["celeba ","./results/multi_clf "]
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    # reproducibility
    torch.manual_seed(777)
    np.random.seed(777)

    # OUT_DIR="../data/relabelled/"
    # OUT_DIR=OUT_DIR+str(args.class_idx)+"/"
    
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    # train_dataset = build_even_celeba_classification_dataset(OUT_DIR,
    #       'train')
    # valid_dataset = build_even_celeba_classification_dataset(OUT_DIR,
    #     'val')
    # test_dataset = build_even_celeba_classification_dataset(OUT_DIR,
    #     'test')
    train_dataset = build_even_celeba_classification_dataset(
  			'train', args.class_idx)
    valid_dataset = build_even_celeba_classification_dataset(
  			'val', args.class_idx)
    test_dataset = build_even_celeba_classification_dataset(
  			'test', args.class_idx)
    n_classes = 2
    CLF_PATH = os.path.join(args.out_dir,str(args.class_idx),"model_best.pth.tar")
    

    f=open("./logs/Attribute_classifier_accuracy.txt","a")
    if args.split_type=="test":
        f.write("Testing on test sample size: "+str(len(test_dataset))+" and attributes "+str(args.class_idx)+"\n")
        print(len(test_dataset))
    
        # train/validation split (Shuffle and batch the datasets)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    else:
        f.write("Testing on test sample size: "+str(len(valid_dataset))+" and attributes "+str(args.class_idx)+"\n")
        print(len(valid_dataset))
    
        # train/validation split (Shuffle and batch the datasets)
        test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False)       

    def test(loader):
        model.eval()
        test_loss = 0
        correct = 0
        num_examples = 0
        predictedArray=[]
        targetArray=[]
        
        #Debug Plot for visualisation
        index=0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data = data.float() / 255.
                target = target.long()

                # run through model
                logits, probas = model(data)
                # print (logits[0])
                # print (target[0])
                # test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
                _, pred = torch.max(probas, 1)
                # num_examples += target.size(0)
                # correct += (pred == target).sum()
                predictedArray.append(pred.cpu().numpy())
                targetArray.append(target.cpu().numpy())
                
                #Debug Plot for visualisation
                hard_Labels=np.argmax(probas.cpu().detach(),axis=1)
                # index=saveImg(data,hard_Labels,index)
                
            predicted=np.concatenate(predictedArray)
            target=np.concatenate(targetArray)
            correct=(predicted==target)
            scores=np.zeros(len(np.unique(target)))
            for index in range(len(np.unique(target))):
                position=np.where(target==index)[0]
                scores[index]=correct[position].sum()/len(position)
                  
        for i in range(len(scores)):
            f.write(" Attribute_%i="%i+str(scores[i]))
        f.write("\n")
        # f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, num_examples,100. * correct / num_examples))
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, num_examples,
        #     100. * correct / num_examples))
        return 0

    # classifier has finished training, evaluate sample diversity
    best_loss = sys.maxsize
    clf_state_dict = torch.load(CLF_PATH)['state_dict']
 
    # reload best model
    model_cls = build_model('celeba')
    # Resnet-18
    model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=n_classes, grayscale=False)
    # Resnet-34 https://www.analyticsvidhya.com/blog/2021/06/build-resnet-from-scratch-with-python/#:~:text=ResNet%20architecture%20uses%20the%20CNN,batchnorm2d%20after%20each%20conv%20layer.&text=Then%20create%20a%20ResNet%20class,and%20the%20number%20of%20classes.
    # model = model_cls(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=n_classes, grayscale=False)
    model = model.to(device)
     # model=Net(n_classes)
     # model.cuda()
    model.load_state_dict(clf_state_dict)

    # get test
    test_loss = test(test_loader)
    f.close()