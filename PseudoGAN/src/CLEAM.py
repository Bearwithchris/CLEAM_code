# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:18:21 2021

"""


# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
import os 
import matplotlib.pyplot as plt 
import copy
import plot as p
from scipy import stats
# from statsmodels.distributions.empirical_distribution import ECDF
from shapely.geometry import Polygon,LineString
import argparse
# prefix="../logs/Real_classifier/N3K/2_point7"


# def plot_CI(prefix,S=30,N=3000,iterrations=50):
#     file=np.load(os.path.join(prefix,"pred_dist.npz"))['x']
#     p0=file[:,0]
#     # S=30
#     # N=3000
#     # iterrations=50
    
#     m=[]
#     up=[]
#     down=[]
    
#     for i in range(iterrations):
#         selected=np.random.choice(p0,S)
#         mu=np.mean(selected)
#         sigma=np.std(selected)
#         delta=1.960*sigma/np.sqrt(S)
#         print ("Mean: {:.5f} CI: {:.5f}".format(mu,delta))
        
#         m.append(mu)
#         up.append(mu+delta)
#         down.append(mu-delta)
    
#     runs=np.arange(0,iterrations,1)
#     fig = plt.figure(figsize=(20, 15))
#     ax = fig.add_subplot(111)   
    
#     ax.plot(runs,m, label="Mean")
#     ax.plot(runs,up,"--",color="orange",label="Upper/lower CI")
#     ax.plot(runs,down,"--",color="orange")
    
#     plt.xlabel("Resample attempts")
#     plt.ylabel(r"$P_\theta$")
#     ax.legend()
#     plt.title("S={},N={} Real $G_0$, Real $C$".format(S,N))
#     plt.rcParams.update({'font.size': 25})
        
def theoretical_mu_sd(at1,at2,N,bias):
    mu=((N*bias*at1)+(N*(1-bias)*(1-at2)))/N
    # sd=np.sqrt(N*bias*at1*(1-at1)+N*(1-bias)*(1-at2)*at2)/N
    pStar0Test=bias
    # sd=np.sqrt((pStar0Test*accAt[0]*(1-pStar0Test*accAt[0])+(1-pStar0Test)*(1-accAt[1])*(1-(1-pStar0Test)*(1-accAt[1])))/N) #Variable p^*
    # print ("Theoretical SD: {:.5f}  Mean: {:.5f} ".format(sd,mu))
    
    #New SD with covariance
    a0=at1 #Alpha 0
    ap0=1-at1 #Alpha prime 0
    a1=at2 #Alpha 1
    ap1=1-at2 #Alpha prime 1
    ps0=bias
    ps1=1-bias
    
    var=(1/N)*((ps0*a0)-(ps0*a0)**2) + (1/N)*((ps1*ap1)+(ps1*ap1)**2) + (2/N)*(ps0*ps1*a0*ap1)
    sd=np.sqrt(var)
    
    return mu,sd

def reverse(at1,at2,mu):
    pstary=(mu-at1)/(1-at1-at2)
    pstarx=1-(pstary)
    return (pstarx,pstary)

# def cal_CI(mu,sigma,S,CI=1.64):
#     up=mu+CI*(sigma/np.sqrt(S))
#     down=mu-CI*(sigma/np.sqrt(S))
#     return up,down

#Not to be used, just testing
# def test_pstar(at1,at2,muE,sigmaE,S):  
#     # at1=0.843943+0.005
#     # at2=0.731096-0.005
    
#     up,down=cal_CI(muE,sigmaE,S)
#     low0,low1=reverse(at1,at2,down)
#     high0,high1=reverse(at1,at2,up)
#     # print ("[{},{}".format(low0,high0))
#     return low0,high0,low1,high1
    
# def plot_approx_scatter(gt0,p0,pStar0,pStarRange0,gen):
    
#     #Plot scatter plot------------------------------------------------------
#     #Raw sample data (p-hat)
#     p1=1-p0
#     plt.scatter(p0,p1,c="red",marker=".",label=r"$\hat{p}$ (Baseline)")
    
#     #Ground Truth
#     if gen==0:
#         gt1=1-gt0
#         plt.plot(gt0,gt1,c="lightblue",marker="D",markersize=12, label="Ground Truth")
    
#     #Approximated value (MLE)
#     pStar1=1-pStar0
#     plt.plot(pStar0,pStar1,c="green",marker="x",markersize=12, label=r"MLE $p*$ (Ours)")
    
#     #p* range
#     pStarRange1=1-pStarRange0
#     pstarRange=np.stack((pStarRange0,pStarRange1),axis=1)
#     polygon1 = LineString(pstarRange.tolist())
#     plt.plot(*polygon1.xy, linewidth=4, c="yellow",alpha=0.5, label=r"$p*$ range (Ours)")
    
#     print(r"$p_0$, S={}, $p^*_0$=0.{}, $\mu_p=${:.3f}, $\sigma_p$={:.5f}".format(len(p0),bias,np.mean(p0),np.std(p0)))
    
#     plt.xlabel(r"$\hat{p}_0$")
#     plt.ylabel(r"$\hat{p}_1$")
#     plt.legend(loc=1,fontsize=13)
#     plt.show()

def plot_samples_problem_illustration(gt0,p0,lower,upper,N,S):
     #Plot raw data for p0
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(221)  
    # ax1.set_title(r"$p_0$, S={}, $p^*_0$=0.{}, $\mu_p=${:.3f}, $\sigma_p$={:.5f}".format(len(p0),bias,np.mean(p0),np.std(p0)))
    
    ax1.set_xlabel(r"$\hat{p}_0$")
    ax1.set_ylabel("Count")
    ax1.hist(p0,alpha=0.5,label=r"$\hat{p}$ (Baseline)")
    ax1.axvline(gt0, color='k', linestyle='dashed', linewidth=2,label=r"Ground truth, $p^*$")
    ax1.axvline(lower, color='g', linestyle='solid', linewidth=2,label=r"$\mathcal{L}(p^*_0)$,$\mathcal{U}(p^*_0)$ (Ours)")
    ax1.axvline(upper, color='g', linestyle='solid', linewidth=2)
    # ax1.hist(gt0,label="Ground truth")
    plt.legend( loc=1 )
    # plt.savefig("../logs/raw_p0_bias_{}.png".format(bias))
    plt.savefig("../logs/{}_p0_N{}_S{}_bias_{}.pdf".format(LA,N,S,gt0),bbox_inches='tight')
    plt.show()
    
    
        
# def plot_samples(p0,gen=0,samples=0):
#      #Plot raw data for p0
#     fig = plt.figure(figsize=(20, 10))
#     ax1 = fig.add_subplot(221)  
#     if gen==1:
#         if samples==0:
#             ax1.set_title(r"S={}, $\mu_p=${:.3f}, $\sigma_p$={:.5f}".format(len(p0),np.mean(p0),np.std(p0)))
#         else:    
#             pass
#     else:
#         ax1.set_title(r"$p_0$, S={}, $p^*_0$=0.{}, $\mu_p=${:.3f}, $\sigma_p$={:.5f}".format(len(p0),bias,np.mean(p0),np.std(p0)))
#     ax1.set_xlabel(r"$\hat{p}_0$")
#     ax1.set_ylabel("Impressions")
#     ax1.hist(p0)
#     # plt.savefig("../logs/raw_p0_bias_{}.png".format(bias))
#     plt.show()



# def run2(bias,accAt,prefix,gen=0,S=30):
        

#         debug=0
#     # 

#         #Emprical data information
#         N=1000
#      #   M=30
#         file=np.load(os.path.join(prefix,"pred_dist.npz"))['x']
#         p0=file[:,0]

#         #Plot the raw S=100 Samples
#         # if gen==0 and debug!=1:
#         #     #Plot raw data for p0
#         #     fig = plt.figure(figsize=(20, 10))
#         #     ax1 = fig.add_subplot(221)  
#         #     ax1.set_title(r"S={}, $p^*_0$=0.{}, $\mu_p=${:.3f}, $\sigma_p$={:.5f}".format(len(p0),bias,np.mean(p0),np.std(p0)))
#         #     ax1.set_xlabel(r"$\hat{p}_0$")
#         #     ax1.set_ylabel("Impressions")
#         #     ax1.hist(p0)
#         #     # plt.savefig("../logs/raw_p0_bias_{}.png".format(bias))
#         #     plt.show()
            
#         rng = np.random.default_rng()
        
#         #Sample S=30 of the data
#         selected=rng.choice(p0,S,replace=False)
#         mup=np.mean(selected)
#         sigmap=np.std(selected)
#         pStar0,pStar1=reverse(accAt[0],accAt[1],mup)
        
#         zalpha=1.96
#         #Calculate approximated CI for pstar
#         lower=(mup-zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))
#         upper=(mup+zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))           
#         print ("Possible range of pStar from samples are [ {:.4f} , {:.4f} ]".format(lower,upper))           
#         if debug!=1: 
#             #Plot raw samples before selection (S=100)
#             # plot_samples(p0,gen)
            
#             #Plot selected data
#             if gen==0:
#                 plot_samples_problem_illustration(bias/10,selected,lower,upper)
#                 # pass
#             else:
#                 plot_samples(selected,gen,1)
                
#             #Plot selected data
#             plot_approx_scatter(bias/10,selected,pStar0,np.array([lower,upper]),gen)
#             mut,stdt=theoretical_mu_sd(accAt[0],accAt[1],N,bias/10.0)
#             print ("Theoretical calculation: Mean={}  Std={}".format(mut,stdt))
            
#         return (selected,pStar0,lower,upper)
    
def CLEAM(accAt,phat,N,S,bias):
        
        # N=1000
        # files=os.listdir(prefix)
        
        #Filer Phat-0 data
        phat0=phat[:,0]
        # bias=float(files[0].strip(".npz").split("_")[-1])
        
        #Calculate statistics
        mup=np.mean(phat0)
        sigmap=np.std(phat0)
        
        #Calculate CLEAM
        pStar0,pStar1=reverse(accAt[0],accAt[1],mup)      
        zalpha=1.96
        #Calculate approximated CI for pstar
        lower=(mup-zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))
        upper=(mup+zalpha*sigmap/np.sqrt(S)-(1-accAt[1]))/(accAt[0]-(1-accAt[1]))           
        

        plot_samples_problem_illustration(bias,phat0,lower,upper,N,S)

                
            #Plot selected data
            # plot_approx_scatter(bias/10,selected,pStar0,np.array([lower,upper]),gen)
            # mut,stdt=theoretical_mu_sd(accAt[0],accAt[1],N,bias/10.0)
            # print ("Theoretical calculation: Mean={}  Std={}".format(mut,stdt))
            
        return (mup,sigmap,pStar0,lower,upper)   
    
def fl2(p0):
    return np.sqrt((p0-0.5)**2+((1-p0)-0.5)**2)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--npzPath', type=str, help='path of the predicted distribution (phat)', default="../data/relabelled/20/pred_dist/S30_N1000_Seed777/")
    parser.add_argument('--acc' , nargs="+", default=[0.981437692125798,0.975407897848191], type=float)
    # parser.add_argument('--bias' , default=0.9, type=float)
    args = parser.parse_args()
    
    #Parameter Extract
    files=os.listdir(args.npzPath)
    S=int(files[0].strip(".npz").split("_")[2].strip("S"))
    N=int(files[0].strip(".npz").split("_")[3].strip("N"))
    attribute=int(files[0].strip(".npz").split("_")[5])
    attr_lookup=np.array(["5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"
    ])
    LA=attr_lookup[attribute]
    
    plt.rcParams.update({'font.size': 16})
    # accAt=[0.9764719791912981,0.9791912981792386] #Gender
    
    accAt=args.acc
    print ("LA Classifier Accuracy:"+ str(accAt))
    


    # count=0
    # if biasSearch:
    #     accepted=False
    #     while (accepted!=True):
    #         try:
    #             selected,MLE0,low,high=run2(bias,accAt,prefix,gen,S)
    #             if low<bias/10<high:
    #                 accepted=True
    #                 print ("Possible range of pStar from samples are [ {:.4f} , {:.4f} ]".format(low,high))
    #             count+=1
    #         except:
    #             count+=1
    #             pass
    #     print ("Count: "+str(count))
    # else:
    #     selected,MLE0,low,high=run2(bias,accAt,prefix,gen,S)
    outLogs=[]
    for i in files:
        bias=float(i.strip(".npz").split("_")[-1])       
        phat=np.load(os.path.join(args.npzPath,i))['x']
        mup,sigmap,MLE0,low,high=CLEAM(accAt,phat,N,S,bias)
        baselineLow=mup-1.96*(sigmap/np.sqrt(S))
        baselineHigh=mup+1.96*(sigmap/np.sqrt(S))
        
        print ("================" + i + "=============")
        print ("Baseline--------------------------------------------")
        print ("Baseline PE of  p*_0= {:.4f} -----> f={:.4f} ".format(mup,fl2(mup)))   
        print ("Baseline IE of p*_0= [ {:.4f} , {:.4f} ] -----> f= [ {:.4f} , {:.4f} ]".format(baselineLow,baselineHigh,fl2(baselineLow),fl2(baselineHigh)))
        baselinePEerror=(abs(mup-bias)/bias)*100
        baselineIEerrorLow=(abs(baselineLow-bias)/bias)*100
        baselineIEerrorHigh=(abs(baselineHigh-bias)/bias)*100
        print ("Baseline Normalised Errors: PE= {:3f} IE= [ {:3f} , {:3f} ]".format(baselinePEerror,baselineIEerrorLow,baselineIEerrorHigh))
        
        
        print ("CLEAM--------------------------------------------")
        print ("CLEAM PE of  p*_0= {:.4f} -----> f={:.4f} ".format(MLE0,fl2(MLE0)))   
        print ("CLEAM IE of p*_0= [ {:.4f} , {:.4f} ] -----> f= [ {:.4f} , {:.4f} ]".format(low,high,fl2(low),fl2(high)))
        CLEAMPEerror=(abs(MLE0-bias)/bias)*100
        CLEAMIEerrorLow=(abs(low-bias)/bias)*100
        CLEAMIEerrorHigh=(abs(high-bias)/bias)*100
        print ("CLEAM Normalised Errors: PE= {:3f} IE= [ {:3f} , {:3f} ]".format(CLEAMPEerror,CLEAMIEerrorLow,CLEAMIEerrorHigh))
        
        
        logs={"pstar":bias,"BasePe":mup,"BaseIe0":baselineLow,"BaseIe1":baselineHigh,
              "BasePeError":baselinePEerror, "BaseIeError0":baselineIEerrorLow, "BaseIeError1":baselineIEerrorHigh,
              "CLEAMPe":MLE0,"CLEAMIe0":low,"CLEAMIe1":high,
              "CLEAMPeError":CLEAMPEerror,"CLEAMIeError0":CLEAMIEerrorLow, "CLEAMIeError1":CLEAMIEerrorHigh
              }
        outLogs.append(logs)
    np.savez("../logs/{}_N_{}_S_{}.npz".format(LA,N,S),x=outLogs)
        # print ("f of p* range: [{},{}]".format(fl2(low),fl2(high)))
    
    
    
        

    