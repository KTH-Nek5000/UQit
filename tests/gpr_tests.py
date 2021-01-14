"""
Tests for gpr
"""
#
import os
import sys
import numpy as np
import math as mt
from matplotlib import pyplot as plt
import torch
import gpytorch
from UQit.gpr_torch import gpr, gprPost, gprPlot
import UQit.analyticTestFuncs as analyticTestFuncs
import UQit.reshaper as reshaper
import UQit.sampling as sampling
#
#
def gprTorch_1d_singleTask_test():
    """
    Test for GPR for 1d parameter
    """
    def fEx(x):
        """
        Simulator
        """
        yEx=np.sin(2*mt.pi*x)
        return yEx

    def noiseGen(n,noiseType):
       """
       Generate a 1D numpy array of standard deviations of the observation noise
       """
       if noiseType=='homo':
          sd=0.2   #standard deviation (Note: non-zero, to avoid instabilities)
          sdV=[sd]*n
          sdV=np.asarray(sdV)
       elif noiseType=='hetero':
          sdMin=0.05
          sdMax=0.55
          sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
       return sdV  

    def trainData(xBound,n,noiseType):
        """
        Create training data D={X,Y}
        """
        x=np.linspace(xBound[0],xBound[1],n)
        sdV=noiseGen(n,noiseType)
        y=fEx(x) + sdV * np.random.randn(n)
        return x,y,sdV
      
    #----- SETTINGS ----------------
    n=120           #number of training samples
    nTest=100       #number of test samples
    xBound=[0.,1]   #parameter range
    #Type of the noise
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    nIter_=800           #number of iterations in optimization of GPR hyperparameters
    lr_   =0.1           #learning rate in the optimization of the hyperparameters
    convPlot_=True       #plot convergence of optimization of the GPR hyperparameters
    #------------------------------------------------
    #(0) Assemble gprOpts dictionary
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_}
    #(1) Generate training and test samples
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
    xTest = np.linspace(xBound[0]-0.2, xBound[1]+.2, nTest) #if numpy is used for training
    #xTest = torch.linspace(xBound[0], xBound[1], nTest)   #if torch is used for training
    #(2) Construct the GPR using the training data
    gpr_=gpr(xTrain=xTrain[:,None],yTrain=yTrain[:,None],noiseV=noiseSdev,
             xTest=xTest[:,None],gprOpts=gprOpts)
    post_f=gpr_.post_f
    post_obs=gpr_.post_y
    #(3) Exact response surface
    fExTest=fEx(xTest)
    #(4) Plot
    pltOpts={'title':'Single-task GPR, 1D parameter, %s-scedastic noise'%noiseType}
    gprPlot(pltOpts).torch1d(post_f,post_obs,xTrain,yTrain,xTest,fExTest)
#
def gprTorch_2d_singleTask_test():
    """
    Test for GPR for 2d input
    """
    ##
    def plot_trainData(n,fSamples,noiseSdev,yTrain):
        """
        Plot the noisy training data which are used in GPR. 
        """
        plt.figure(figsize=(10,5))
        x_=np.zeros(n)
        for i in range(n):
            x_[i]=i+1
        for i in range(500):  #only for plottig possible realizations
            noise_=noiseSdev*np.random.randn(n)
            plt.plot(x_,fSamples+noise_,'.',color='steelblue',alpha=0.4,markersize=1)
        plt.errorbar(x_,fSamples,yerr=1.96*abs(noiseSdev),ls='none',capsize=5,ecolor='k',
                elinewidth=4,label=r'$95\%$ CI in Obs.')
        plt.plot(x_,fSamples,'o' ,markersize=6,markerfacecolor='lime',
                markeredgecolor='salmon',label='Mean Observation')
        plt.plot(x_,yTrain ,'xr' ,markersize=6,label='Sample Observation')
        plt.legend(loc='best',fontsize=15)
        plt.ylabel('QoI',fontsize=17)
        plt.xlabel('Simulation Index',fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)    
        plt.title('Training data with associated confidence')
        plt.show()
    ##
    def trainDataGen(p,sampleType,n,qBound,fExName,noiseType):
        """
        Generate Training Data
        """
        #  (a) xTrain 
        if sampleType=='grid': 
          nSamp=n[0]*n[1]
          gridList=[]
          for i in range(p):
              #grid_=torch.linspace(qBound[i][0],qBound[i][1],n[i])   #torch
              grid_=np.linspace(qBound[i][0],qBound[i][1],n[i])
              gridList.append(grid_)
          xTrain=reshaper.vecs2grid(gridList)
#       xTrain = gpytorch.utils.grid.create_data_from_grid(gridList)  #torch
        elif sampleType=='random': 
             nSamp=n     #number of random samples   
             xTrain=sampling.LHS_sampling(n,qBound)
        #  (b) Observation noise   
        #noiseSdev=torch.ones(nTot).mul(0.1)    #torch
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
        #yTrain = torch.sin(mt.pi*xTrain[:,0])*torch.cos(.25*mt.pi*xTrain[:,1])+
        #         torch.randn_like(xTrain[:,0]).mul(0.1)   #torch
        #  (c) Training response
        yTrain=analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val
        yTrain_noiseFree=yTrain
        yTrain=yTrain_noiseFree+noiseSdev*np.random.randn(nSamp)
        return xTrain,yTrain,noiseSdev,yTrain_noiseFree
    ##    
    def noiseGen(n,noiseType,xTrain,fExName):
       """
       Generate a 1D numpy array of standard deviations of the observation noise
       """
       if noiseType=='homo':
          sd=0.2   # noise standard deviation  (Note: non-zero, to avoid instabilities)
          sdV=sd*np.ones(n)
       elif noiseType=='hetero':
          #sdMin=0.01
          #sdMax=0.5
          #sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
          #sdV=0.15*np.ones(n)
          sdV=0.1*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV
    #
    #----- SETTINGS
    qBound=[[-2,2],[-2,2]]   #Admissible range of parameters
    fExName='type1'          #Type of simulator in analyticTestFuncs.fEx2D
                             #'type1', 'type2', 'type3', 'Rosenbrock'
    sampleType='random'      #'random' or 'grid': type of training samples
    if sampleType=='grid':
       n=[9,9]               #number of training samples in each input dimension
    elif sampleType=='random':
       n=100                 #total number of training samples drawn randomly
    noiseType='hetero'       #noise type: 'homo'=homoscedastic, 'hetero'=heterscedastic
    #options for GPR
    nIter_=1000        #number of iterations in optimization of GPR hyperparameters
    lr_   =0.05        #learning rate in the optimization of the hyperparameters
    convPlot_=True     #plot convergence of optimization of GPR hyperparameters
    nTest=[21,20]     #number of test points in each parameter dimension
    #------------------------------------------------
    #(0) Assemble the gprOpts dict
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_}
    #(1) Generate training data
    p=len(qBound)    #dimension of the input
    xTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,sampleType,n,qBound,fExName,noiseType)
    nSamp=yTrain.shape[0]
    plot_trainData(nSamp,yTrain_noiseFree,noiseSdev,yTrain)
    #(2) Create the test samples
    xTestList=[]
    for i in range(p):
        #grid_=torch.linspace(qBound[i][0],qBound[i][1],20)    #torch
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        xTestList.append(grid_)
    xTest=reshaper.vecs2grid(xTestList)
    #(3) Construct the GPR based on the training data and make predictions at the test samples
    gpr_=gpr(xTrain,yTrain[:,None],noiseSdev,xTest,gprOpts)
    post_f=gpr_.post_f
    post_obs=gpr_.post_y
    # Predicted mean and variance of the posteriors at the test grid    
    fP_=gprPost(post_f,nTest)
    fP_.torchPost()
    post_f_mean=fP_.mean
    post_f_sdev=fP_.sdev
    lower_f=fP_.ciL
    upper_f=fP_.ciU
    obsP_=gprPost(post_obs,nTest)
    obsP_.torchPost()
    post_obs_mean=obsP_.mean
    post_obs_sdev=obsP_.sdev
    lower_obs=obsP_.ciL
    upper_obs=obsP_.ciU
    # Plots
    with torch.no_grad():
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(141)        
        fEx_test=analyticTestFuncs.fEx2D(xTest[:,0],xTest[:,1],fExName,'comp').val
        CS0=ax.contour(xTestList[0],xTestList[1],fEx_test.reshape((nTest[0],nTest[1]),order='F').T,levels=40)
        ax.clabel(CS0, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Exact $f(q)$')
        ax = fig.add_subplot(142)
        CS1=ax.contour(xTestList[0],xTestList[1],(post_f_mean).T,levels=40)
        ax.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Mean Posterior of $f(q)$')
        ax = fig.add_subplot(143)
        CS2=ax.contour(xTestList[0],xTestList[1],upper_obs.T,levels=40)
        ax.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Upper Confidence for Observations')
        ax = fig.add_subplot(144)
        CS2=ax.contour(xTestList[0],xTestList[1],lower_obs.T,levels=40)
        ax.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Lower Confidence for Observations')
        plt.show()
        #2dplot
        pltOpts={'title':'Mean posterior of f(q)',
                 'xlab':r'$q_1$',
                 'ylab':r'$q_2$'}
        gprPlot(pltOpts).torch2d_2dcont(xTrain,xTestList,post_f_mean) 
        #3d plot
        gprPlot().torch2d_3dSurf(xTrain,yTrain,xTestList,post_obs)
#
def gprTorch_2d_singleTask_test2():
    """
    Test for GPR for 2d input - Importance of standardization of the training data
    Run the test with 'standardizeYTrain_' being True and False. 
    """
    ##
    def trainData():
        """
        Generate training data
        """
        qBound=[[-1,1],[-1,1]]
        x1_=sampling.trainSample(sampleType='GQ',GQdistType='Unif',qInfo=qBound[0],nSamp=4)
        x2_=sampling.trainSample(sampleType='GQ',GQdistType='Unif',qInfo=qBound[1],nSamp=4)
        xTrain=reshaper.vecs2grid([x1_.q,x2_.q])
        yTrain_mean=np.asarray([-0.0169906,-0.0191095,-0.0167435,-0.0172338,-0.0203195,-0.020089,
        -0.0184691,-0.0188843,-0.0164581,-0.0200013,-0.0186512,-0.0159343,
        -0.0185975, -0.0155899,-0.0178921,-0.018329 ])
        yTrain_sdev= np.asarray([0.00131249,0.00104324,0.00085491,0.00099751,0.00094231,0.00102579,
        0.0010804,0.00089567,0.00081245,0.0011208,0.00110756,0.00126673,
        0.00108875,0.00145115,0.00098541,0.00130559])
        
        return qBound,xTrain,yTrain_mean,yTrain_sdev
    #   
    #----- SETTINGS
    #options for GPR
    nIter_=3000        #number of iterations in optimization of GPR hyperparameters
    lr_   =0.05        #learning rate in the optimization of the hyperparameters
    convPlot_=True     #plot convergence of optimization of GPR hyperparameters
    standardizeYTrain_=True   #standardize the Y training data?
    nTest=[41,40]      #number of test points in each parameter dimension    
    #---------------------------------
    #(1) Generate training data
    qBound,xTrain,yTrain_mean,yTrain_sdev=trainData()
    p=len(qBound)    #dimension of the input
    nSamp=len(yTrain_mean)
    
    #(2) Generate noisy training data
    noise_=np.random.randn(nSamp)
    noise_=yTrain_sdev*noise_
    yTrain=yTrain_mean+noise_

    #(3) Create test points
    xTestList=[]
    for i in range(p):
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        xTestList.append(grid_)
    xTest=reshaper.vecs2grid(xTestList)

    #(4) Fit the GPR
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_,'standardizeYTrain':standardizeYTrain_}
    gpr_=gpr(xTrain,yTrain[:,None],yTrain_sdev,xTest,gprOpts)
    post_f=gpr_.post_f
    post_obs=gpr_.post_y
    #(4) Predicted mean and variance of the posteriors at the test grid
    shift_=0.0 #default shift and scaling, assuming no standardization in the training data
    scale_=1.0
    if standardizeYTrain_:
        shift_=gpr_.shift[0]  #0: single-response
        scale_=gpr_.scale[0]

    fP_=gprPost(post_f,nTest,shift=shift_,scale=scale_)
    fP_.torchPost()
    post_f_mean=fP_.mean
    post_f_sdev=fP_.sdev
    obsP_=gprPost(post_obs,nTest,shift=shift_,scale=scale_)
    obsP_.torchPost()
    post_obs_mean=obsP_.mean
    post_obs_sdev=obsP_.sdev

    # (5) Plots
    # When using 'torch2d_2dcont', de-standardization of the predicted  mean and sdev is manual
    pltOpts={'title':'Mean of posterior f(q)','xlab':r'$q_1$','ylab':r'$q_2$'}
    gprPlot(pltOpts).torch2d_2dcont(xTrain,xTestList,post_f_mean*scale_+shift_)
    pltOpts={'title':'Sdev of posterior f(q)','xlab':r'$q_1$','ylab':r'$q_2$'}
    gprPlot(pltOpts).torch2d_2dcont(xTrain,xTestList,post_f_sdev*scale_)
    # When using torch2d_3dSurf, the optional arguments shift and scale can be passed. 
    # Therefore, de-standardization of the predictions is automatic. 
    pltOpts={'title':'Mean of posterior f(q)','xlab':r'$q_1$','ylab':r'$q_2$'}
    gprPlot(pltOpts).torch2d_3dSurf(xTrain,yTrain,xTestList,post_f,shift=shift_,scale=scale_)
    pltOpts={'title':'Mean of posterior predictive y=f(q)+e','xlab':r'$q_1$','ylab':r'$q_2$'}
    gprPlot(pltOpts).torch2d_3dSurf(xTrain,yTrain,xTestList,post_obs,shift=shift_,scale=scale_)
#    
