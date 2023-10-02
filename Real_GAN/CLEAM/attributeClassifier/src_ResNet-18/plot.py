# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:23:19 2021

"""

import matplotlib.pyplot as plt
import scipy
import os
import numpy as np
import utils
import seaborn as sns
import pandas as pd
import scipy.stats as stats


def plot_fvAcc_exp_single(metric,ibias):
    # metric="L1"
    k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    cAcc_array=[0.9,0.8,0.7]
    # cAcc=0.9 #classifier's accuracy (independent variable)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)  
    
    data_array=np.zeros([100,3])
    for index in range(len(cAcc_array)):
        file=np.load("bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,cAcc_array[index],k))[metric]
        data_array[:,index]=file
        # ax.scatter(np.array([cAcc for i in range(100)]),file)
        ax.set_xticklabels(cAcc_array) 
        print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric,k,cAcc_array[index],ibias,np.around(np.mean(file),4),np.around(np.std(file),4)) )
    
    plt.rcParams.update({'font.size': 20})
    plt.title(metric+" ,$f$ vs $Acc_{avg}$ at bias= "+str(ibias))
    ax.set_ylabel("$f$")
    ax.set_xlabel("$Acc_{avg}$")
    ax.boxplot(data_array)
    plt.savefig("f_vs_acc_at_bias_{}.png".format(ibias))

def plot_fvAcc_exp_multi(k,ibias,prefix="./"):
    metric_array=["L1","L2","Is","Sp"]
    # k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    if prefix=="./sd2":
        cAcc_array=[0.8,0.7,0.6]
    elif prefix=="./sd1":
        cAcc_array=[0.9,0.8,0.7,0.6]
    else:
        cAcc_array=[0.9,0.8,0.7]
        
    # cAcc=0.9 #classifier's accuracy (independent variable)
    
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(144)  
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(142)
    ax4 = fig.add_subplot(141)  
    
    data_array_L1=np.zeros([100,len(cAcc_array)])
    data_array_L2=np.zeros([100,len(cAcc_array)])
    data_array_Is=np.zeros([100,len(cAcc_array)])
    data_array_Sp=np.zeros([100,len(cAcc_array)])
    
    counter=0
    for index in cAcc_array:
        file_L1=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[0]]
        file_L2=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[1]]
        file_Is=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[2]]
        file_Sp=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[3]]
        
        
        data_array_L1[:,counter]=file_L1
        data_array_L2[:,counter]=file_L2
        data_array_Is[:,counter]=file_Is
        data_array_Sp[:,counter]=file_Sp        
        counter+=1
        
        try:
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[0],k,index,ibias,np.around(np.mean(file_L1),4),np.around(np.std(file_L1),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[1],k,index,ibias,np.around(np.mean(file_L2),4),np.around(np.std(file_L2),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[2],k,index,ibias,np.around(np.mean(file_Is),4),np.around(np.std(file_Is),4)) )
            print ("Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[3],k,index,ibias,np.around(np.mean(file_Sp),4),np.around(np.std(file_Sp),4)) )
        except:
            print("error")
                
    plt.rcParams.update({'font.size': 14})        
    ax1.set_xticklabels(cAcc_array) 
    ax2.set_xticklabels(cAcc_array) 
    ax3.set_xticklabels(cAcc_array) 
    ax4.set_xticklabels(cAcc_array) 
    ax4.set_ylabel("$f$")
    
    ax4.boxplot(data_array_L1)
    ax3.boxplot(data_array_L2)
    ax2.boxplot(data_array_Is)
    ax1.boxplot(data_array_Sp)
    
    ax4.title.set_text('L1')
    ax3.title.set_text('L2')
    ax2.title.set_text('Is')
    ax1.title.set_text('$\Delta sp$')
    

    plt.savefig(os.path.join(prefix,"overall_f_vs_acc_at_bias_{}.png".format(ibias)))
    
def plot_fvAcc_exp_multi_deltaf(k,ibias,prefix="./"):
    metric_array=["L1","L2","Is","Sp"]
    # k=4 #Cardinality of attributes (constant)
    # ibias=0.8 #initialise bias (constant)
    if prefix=="./sd2":
        cAcc_array=[0.8,0.7,0.6]
    elif prefix=="./sd1":
        cAcc_array=[0.9,0.8,0.7,0.6]
    else:
        cAcc_array=[0.9,0.8,0.7]
        
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(144)  
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(142)
    ax4 = fig.add_subplot(141)  
    
    data_array_L1=np.zeros([100,len(cAcc_array)])
    data_array_L2=np.zeros([100,len(cAcc_array)])
    data_array_Is=np.zeros([100,len(cAcc_array)])
    data_array_Sp=np.zeros([100,len(cAcc_array)])
    
    counter=0
    for index in cAcc_array:
        fstar=utils.ideal_f(k,index,n=1)
        file_L1=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[0]])
        file_L2=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[1]])
        file_Is=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[2]])
        file_Sp=abs(fstar[0]-np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric_array[3]])
        
        
        data_array_L1[:,counter]=file_L1
        data_array_L2[:,counter]=file_L2
        data_array_Is[:,counter]=file_Is
        data_array_Sp[:,counter]=file_Sp        
        counter+=1
        
        try:
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[0],k,index,ibias,np.around(np.mean(file_L1),4),np.around(np.std(file_L1),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[1],k,index,ibias,np.around(np.mean(file_L2),4),np.around(np.std(file_L2),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[2],k,index,ibias,np.around(np.mean(file_Is),4),np.around(np.std(file_Is),4)) )
            print ("Delta f, Metric={}, k={}, Accuracy={}, Bias={} -> mean={}, sd={}".format(metric_array[3],k,index,ibias,np.around(np.mean(file_Sp),4),np.around(np.std(file_Sp),4)) )
        except:
            print("error")
                
    plt.rcParams.update({'font.size': 14})        
    ax1.set_xticklabels(cAcc_array) 
    ax2.set_xticklabels(cAcc_array) 
    ax3.set_xticklabels(cAcc_array) 
    ax4.set_xticklabels(cAcc_array) 
    ax4.set_ylabel("$\Delta f$")
    
    ax4.boxplot(data_array_L1)
    ax3.boxplot(data_array_L2)
    ax2.boxplot(data_array_Is)
    ax1.boxplot(data_array_Sp)
    
    ax4.title.set_text('L1')
    ax3.title.set_text('L2')
    ax2.title.set_text('Is')
    ax1.title.set_text('$\Delta sp$')
    

    plt.savefig(os.path.join(prefix,"overall_Delta f_vs_acc_at_bias_{}.png".format(ibias)))


#Plot the impression vs f score graphs 
def plot_density_plot(metric,k,ibias,prefix="./"):
    cAcc_array=[0.9,0.8,0.7]
    # metric_array=["L1","L2","Is","Sp"]
    
    df=pd.DataFrame(columns=["data","labels"])
    
    for index in cAcc_array:
        file=np.load(os.path.join(prefix,"bias_{}_Acc_{}_fScores_k_{}.npz".format(ibias,index,k)))[metric]
        cAcc_label=np.ones_like(file)*index
        df_temp=pd.DataFrame({"data":file,"labels":cAcc_label})
        df=pd.concat([df,df_temp])
        
    sns.kdeplot(data=df, x="data", hue="labels")
    plt.xlabel("$f \, score$")
    plt.savefig(os.path.join(prefix,"{}_bias_{}_k_{}.png".format(metric,ibias,k)))
    

def scatter_plot_p(lSamples,ibias,cAcc,k,prefix="./"):
    file=np.load(os.path.join(prefix,"lsample_{}_p_bias_{}_Acc_{}_k_{}.npz".format(lSamples,ibias,cAcc,k)))['x']
    plt.scatter(file[:,0],file[:,1])
    plt.title("N=%i, $p_{bias}$=%f, $Acc_{avg}$=%f, $k$=%i"%(lSamples,np.around(ibias,2),np.around(cAcc,2),k))
    roundV=5
    print ("Average p measurement= [{},{}]".format(np.around(np.mean(file[:,0]),roundV),np.around(np.mean(file[:,1]),roundV)))
    print ("Standard deviation in P=[{},{}]".format(np.around(np.std(file[:,0]),roundV),np.around(np.std(file[:,1]),roundV)))
    plt.xlabel("$p_0$")
    plt.ylabel("$p_1$")
    plt.show()
    
def histogram_plot_p(lSamples,ibias,cAcc,accAt,k,gen=0,prefix="./"):
    if gen==0:
        file=np.load(os.path.join(prefix,"lsample_{}_p_bias_{}_Acc_{}_k_{}.npz".format(lSamples,ibias,cAcc,k)))['x']
    else:
        file=np.load(os.path.join(prefix,"pred_dist.npz"))['x']
    bins_num=20
    p0=file[:,0]
    mu=np.mean(p0)
    sigma=np.std(p0)
    plt.hist(p0,bins=bins_num)
    if gen==0:
        plt.title("N=%i, $p_{bias}$=%f, $Acc_{avg}$=%f, $k$=%i, $\mu=$%f,  $\sigma$=%f"%(lSamples,np.around(ibias,2),np.around(cAcc,2),k,mu,sigma))
        mu,sd=theoretical_mu_sd(accAt[0],accAt[1],lSamples,ibias)
        x_axis=np.linspace(mu - 3*sd, mu + 3*sd, lSamples)
        y_axis=stats.norm.pdf(x_axis, mu, sd)
        plt.plot(x_axis,y_axis,label="Theoretical")
        
        plt.legend()
        
    else:
        #For gen_dist
        plt.title("N=%i, $Acc_{avg}$=%f, $k$=%i, $\mu=$%f,  $\sigma$=%f"%(lSamples,np.around(cAcc,2),k,mu,sigma))
    plt.xlabel("$p_0$")
    plt.ylabel("$Impressions$")
    
    
def theoretical_mu_sd(at1,at2,N,bias):
    mu=((N*bias*at1)+(N*(1-bias)*(1-at2)))/N
    sd=np.sqrt(N*bias*at1*(1-at1)+N*(1-bias)*(1-at2)*at2)/N
    print ("Theoretical Mean: {:.5f} SD: {:.5f}   ".format(mu,sd))
    return mu,sd

def reverse(at1,at2,mu):
    pstary=(mu-at1)/(1-at1-at2)
    pstarx=1-(pstary)
    return (pstarx,pstary)
    
    
def empirical_vs_theoretical_overlap(lSamples,ibias,cAcc,accAt,k,prefix="./",pmode=0):
    #Empirical
    file=np.load(os.path.join(prefix,"pred_dist.npz"))['x']
    bins_num=80
    # bins_num=20
    p0=file[:,pmode]
    mu=np.mean(p0)
    sigma=np.std(p0)
    plt.hist(p0,bins=bins_num,label="Empirical(Trained $C$)")
    
    #Theoretetical
    mu,sd=theoretical_mu_sd(accAt[0],accAt[1],lSamples,ibias)
    x_axis=np.linspace(mu - 3*sd, mu + 3*sd, lSamples)
    y_axis=stats.norm.pdf(x_axis, mu, sd)
    plt.plot(x_axis,y_axis,label="Theoretical")
    
    plt.legend()
    plt.title("N=%i, $p_{bias}$=%f, $Acc_{avg}$=%f, $k$=%i, $\mu=$%f,  $\sigma$=%f"%(lSamples,np.around(ibias,2),np.around(cAcc,2),k,mu,sigma))
    if pmode==0:
        plt.xlabel("$p_0$")
    else:
        plt.xlabel("$p_1$")
    plt.ylabel("$Impressions$")
    
def plot_sample_vs_theoretical(pstar,mue,sde,mut,sdt,S):

    #Empirical
    xe=np.linspace(mue - 3*sde, mue + 3*sde, S)
    ye=stats.norm.pdf(xe, mue, sde)
    plt.plot(xe,ye,label="Empirical")

    #Theoretical
    xt=np.linspace(mut - 3*sdt, mut + 3*sdt, S)
    yt=stats.norm.pdf(xt, mut, sdt)
    plt.plot(xt,yt,label="Theoretical")
    
    plt.title(r"$p^*=${:5f}, $\mu_T=${:5f}, $\sigma_T={:6f}$".format(pstar,mut,sdt))
    
    plt.legend()
    
    

if __name__=="__main__":
    #Real classifier----------------------------------------------------------------------
    # metric="L2"
    k=2
    ibias=0.90
    accAt=[0.90,1.0]
    # accAt=[0.7810173046670162,0.8076560041950708] #<=1.5k
    # accAt=[0.756581017304667,0.83209229155742] #3k
    
    
    
    #"Fake" classifier---------------------------------------------------------------------
    # u_acc_array_k2=[[0.9,0.9],[0.8,0.8],[0.7,0.7],[0.6,0.6],[0.5,0.5]]
    # u_acc_array_k2_sd1=[[1.0,0.8],[0.9,0.7],[0.8,0.6],[0.7,0.5]]
    # accAt=u_acc_array_k2[3]
    
    cAcc=sum(accAt)/2
    lSamples=3000
    # empirical_vs_theoretical_overlap(lSamples,ibias,cAcc,accAt,k,prefix="../logs/Real_classifier/N3K/39_point9",pmode=0)
    
    histogram_plot_p(lSamples,ibias,cAcc,accAt,k,gen=0,prefix="./")
    
    # histogram_plot_p(lSamples,ibias,cAcc,accAt,k,gen=0,prefix="./")
    
    # plot_density_plot(metric,k,0.6,prefix="./k=2/0.6/uniform")
    # ibias=0.9 #initialise bias (constant)
    # # plot_fvAcc_exp_single(metric,ibias)
    # prefix=""
    # plot_fvAcc_exp_multi(k,ibias,prefix)
    # plot_fvAcc_exp_multi_deltaf(k,ibias,prefix)