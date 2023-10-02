# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:41:26 2022
"""

import matplotlib.pyplot as plt
import numpy as np


controlSize= 50
classidx=2#20



bias_array=np.linspace(0.9,0.5,5)
divMean_array=[]
divStd_array=[]
trueDiv_array=[]


for bias in bias_array:
    TrueD=bias-(1-bias)
    data=np.load("./logs/class{}_bias{}_controlsize{}.npz".format(classidx,int(bias*10),controlSize))['x']
    divMean_array.append(np.mean(data))
    divStd_array.append(np.std(data))
    trueDiv_array.append(TrueD)
    
fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.errorbar(bias_array,np.array(divMean_array), fmt=".",yerr=np.array(divStd_array),capsize=3, label="Proxy DI score w. control set")
ax.plot(bias_array,trueDiv_array,linestyle='dashed',color="0",label="Ground Truth Diversity Score")
ax.set_ylabel("Diversity score", fontsize=10)
ax.set_xlabel(r"Ground Truth, $p^*_0$", fontsize=10)
ax.grid(color='gray', linestyle='dashed')
plt.legend()
plt.savefig("./logs/Approximation_class{}_bias{}_controlsize{}.pdf".format(classidx,int(bias*10),controlSize),bbox_inches='tight')
plt.show()


#New plot
fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.plot(bias_array,abs(np.array(divMean_array)-np.array(trueDiv_array)),linestyle='-',color="0",label="Error of Proxy DI score w. control set")
ax.legend()
ax.set_ylabel("Error from GT", fontsize=10)
ax.set_xlabel(r"Ground Truth, $p^*_0$", fontsize=10)
plt.savefig("./logs/Error_class{}_bias{}_controlsize{}.pdf".format(classidx,int(bias*10),controlSize),bbox_inches='tight')
plt.show()

def plot_gender_blackhair():
    bias_array=np.linspace(0.5,0.9,5)
    divMean_array=[]
    divStd_array=[]
    trueDiv_array=[]
    
    
    for bias in bias_array:
        TrueD=bias-(1-bias)
        data=np.load("./logs/class{}_bias{}_controlsize{}.npz".format(20,int(bias*10),controlSize))['x']
        divMean_array.append(np.mean(data))
        divStd_array.append(np.std(data))
        trueDiv_array.append(TrueD)
        
    fig = plt.figure(figsize=(14, 3))
    ax = fig.add_subplot(121)
    ax.errorbar(bias_array,np.array(divMean_array), fmt=".",yerr=np.array(divStd_array),capsize=3, label="Proxy DI Score W. Control Set")
    ax.plot(bias_array,trueDiv_array,linestyle='dashed',color="0",label="Ground Truth Diversity Score")
    ax.grid(color='gray', linestyle='dashed')
    
    # plt.setp(ax.get_xticklabels(), visible=False)
    
    plt.legend(loc=0,fontsize=12)
    
    divMean_array=[]
    divStd_array=[]
    trueDiv_array=[]
    
    for bias in bias_array:
        TrueD=bias-(1-bias)
        data=np.load("./logs/class{}_bias{}_controlsize{}.npz".format(8,int(bias*10),controlSize))['x']
        divMean_array.append(np.mean(data))
        divStd_array.append(np.std(data))
        trueDiv_array.append(TrueD)
        
    ax2 = fig.add_subplot(122)
    ax2.errorbar(bias_array,np.array(divMean_array), fmt=".",yerr=np.array(divStd_array),capsize=3, label="Proxy DI score w. control set")
    ax2.plot(bias_array,trueDiv_array,linestyle='dashed',color="0",label="GT Diversity Score")
      
    
    
    ax.set_ylabel("Diversity score", fontsize=10)
    ax.set_xlabel(r"Ground Truth, $p^*_0$", fontsize=10)
    ax2.set_ylabel("Diversity score", fontsize=10)
    ax2.set_xlabel(r"Ground Truth, $p^*_0$", fontsize=10)
    
    
    ax2.grid(color='gray', linestyle='dashed')
    # plt.savefig("./logs/Approximation_class{}_bias{}_controlsize{}.pdf".format(classidx,int(bias*10),controlSize),bbox_inches='tight')
    plt.subplots_adjust(wspace=0.15)
    ax.set_ylim(bottom=-0.4,top=1.0)
    ax2.set_ylim(bottom=-0.4,top=1.0)
    plt.savefig("./logs/Approximation_class20_9_controlsize{}.pdf".format(controlSize),bbox_inches='tight')
    plt.show()
    
# plot_gender_blackhair()