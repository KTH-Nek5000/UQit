"""
Tests for ppce
"""
import os
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from ppce import ppce
import pce
import gpr_torch
import statsUQit
import analyticTestFuncs
import writeUQ
import reshaper
import sampling
#
def ppce_1d_test():
    """
    Test PPCE over 1D parameter space
    """
    def fEx(x,fType,qInfo):
        """
        Simulator
        """
        yEx=analyticTestFuncs.fEx1D(x,fType,qInfo).val
        return yEx
    #
    def noiseGen(n,noiseType):
        """
        Generate a 1D numpy array of the standard deviation of the observation noise
        """
        if noiseType=='homo': #homoscedastic noise
           sd=0.1   #(non-zero, to avoid instabilities)
           sdV=[sd]*n
           sdV=np.asarray(sdV)
        elif noiseType=='hetero': #heteroscedastic noise
           sdMin=0.02
           sdMax=0.2
           sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
        return sdV  
    #
    def trainData(xInfo,n,noiseType,trainSamplyType,distType,fType):
        """
        Create training data D={X,Y}
        """
        X_=sampling.trainSample(sampleType=trainSampleType,GQdistType=distType,qInfo=xInfo,nSamp=n)
        x=X_.q
        sdV=noiseGen(n,noiseType)
        y=fEx(x,fType,xInfo) + sdV * np.random.randn(n)
        return x,y,sdV
    #
    #-------SETTINGS------------------------------
    distType='Norm'      #type of distribution of the parameter (Acc. gPCE rule)
    trainSampleType='normRand'   #how to draw the trainining samples, see trainSample in sampling.py
    qInfo=[0.5,0.9]     #info about the parameter
                    #if 'Unif', qInfo =[min(q),max(q)]
                    #if 'Norm', qInfo=[m,v] for q~N(m,v^2)
    n=30            #number of training samples in GPR
    noiseType='homo'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    nGQtest=10         #number of test points (=Gauss quadrature nodes)
    #GPR options
    nIter_gpr=1000      #number of iterations in optimization of hyperparameters
    lr_gpr   =0.2       #learning rate for the optimization of the hyperparameters
    convPlot_gpr=True  #plot convergence of the optimization of the GPR hyperparameters
    nMC=1000           #number of samples drawn from GPR surrogate to construct estimates 
                       #  for the moments of f(q)
    #---------------------------------------------
    if distType=='Unif':
       fType='type1'
    elif distType=='Norm':
       fType='type2'
    #(1) generate synthetic training data
    qTrain,yTrain,noiseSdev=trainData(qInfo,n,noiseType,trainSampleType,distType,fType)
    #(2) assemble the ppceDict dictionary
    ppceDict={'nGQtest':nGQtest,'qInfo':qInfo,'distType':distType,
              'nIter_gpr':nIter_gpr,'lr_gpr':lr_gpr,'convPlot_gpr':convPlot_gpr,'nMC':nMC}
    #(3) construct the ppce
    ppce_=ppce(qTrain,yTrain,noiseSdev,ppceDict)
    fMean_samples=ppce_.fMean_samps
    fVar_samples=ppce_.fVar_samps
    optOut=ppce_.optOut
    #(4) postprocess
    #   (a) plot the GPR surrogate along with response from the exact simulator
    pltOpts={'title':'PPCE, 1D param, %s-scedastic noise'%noiseType}
    gpr_torch.gprPlot(pltOpts).torch1d(optOut['post_f'],optOut['post_obs'],qTrain,yTrain,
            optOut['qTest'][0],fEx(optOut['qTest'][0],fType,qInfo))
    #   (b) plot histogram and pdf of the mean and variance distribution
    statsUQit.pdfFit_uniVar(fMean_samples,True,[])
    statsUQit.pdfFit_uniVar(fVar_samples,True,[])
    #   (c) compare the exact moments with estimated values by ppce
    fEx=analyticTestFuncs.fEx1D(qTrain,fType,qInfo)
    fEx.moments(qInfo)
    fMean_ex=fEx.mean
    fVar_ex=fEx.var

    fMean_mean=fMean_samples.mean()
    fMean_sdev=fMean_samples.std()
    fVar_mean=fVar_samples.mean()
    fVar_sdev=fVar_samples.std()
    print(writeUQ.printRepeated('-', 80))
    print('>> Exact mean(f) = %g' %fMean_ex)
    print('   ppce estimated: E[mean(f)] = %g , sdev[mean(f)] = %g' %(fMean_mean,fMean_sdev))
    print('>> Exact Var(f) = %g' %fVar_ex)
    print('   ppce estimated: E[Var(f)] = %g , sdev[Var(f)] = %g' %(fVar_mean,fVar_sdev))
#
def ppce_2d_test():
    """
    Test for ppce for 2D parameter
    """
    def fEx(p,sampleType,n,qInfo,fExName):
        """
        Generate synthetic training data
        """
        #  (a) xTrain
        nSamp=n[0]*n[1]
        xi=[]
        q=[]
        qBound=[]
        if sampleType[0]=='LHS' and sampleType[1]=='LHS':
           if distType==['Unif']*p:
              qBound=qInfo
              xi=sampling.LHS_sampling(nSamp,[[-1,1]]*p)
              xTrain=np.zeros((nSamp,p))
              for i in range(p):
                  xTrain[:,i]=pce.pce.mapFromUnit(xi[:,i],qBound[i])
              yTrain=analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val
           else:
              raise ValueError("LHS works only when all q have 'Unif' distribution.")
        else:
           for i in range(p):
               samps=sampling.trainSample(sampleType=sampleType[i],GQdistType=distType[i],
                      qInfo=qInfo[i],nSamp=n[i])
               q.append(samps.q)
           xTrain=reshaper.vecs2grid(q)
           yTrain=analyticTestFuncs.fEx2D(q[0],q[1],fExName,'tensorProd').val
        return xTrain,yTrain
    #
    def trainDataGen(p,sampleType,n,qInfo,fExName,noiseType):
        """
        Generate synthetic training data
        """
        #  (a) xTrain and noise-free yTrain
        xTrain,yTrain_noiseFree=fEx(p,sampleType,n,qInfo,fExName)
        nSamp=xTrain.shape[0]
        #  (b) set the sdev of the observation noise
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
        #  (c) Training data
        yTrain=yTrain_noiseFree+noiseSdev*np.random.randn(nSamp)
        return xTrain,yTrain,noiseSdev,yTrain_noiseFree
    #
    def noiseGen(n,noiseType,xTrain,fExName):
       """
       Generate a 1D numpy array of standard deviation of the observation noise
       """
       if noiseType=='homo':
          sd=0.2   #(non-zero, to avoid instabilities)
          sdV=sd*np.ones(n)
       elif noiseType=='hetero':
          sdV=0.1*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV  
    #
    #----- SETTINGS -------------------------------------------
    #settings for parameters and data
    qInfo=[[-2,2],[-2,3]]    #info about the parameter
                             #if 'Unif', qInfo =[min(q),max(q)]
                             #if 'Norm', qInfo=[m,v] for q~N(m,v^2)
    distType=['Unif','Unif']    #distribution type of parameters
    fExName='type1'          #name of simulator to generate synthetic dat
                             #see analyticTestFuncs.fEx2D()
    trainSampleType=['LHS','LHS']   #sampling type, see trainSample in sampling.py
    n=[10,12]               #number of training samples for each parameter.
                            #note: n[0]*n[1]<128, due to GpyTorch
    noiseType='hetero'      #type of observation noise
                            #'homo'=homoscedastic, 'hetero'=heterscedastic
    #options for GPR
    nIter_gpr=1000      #number of iterations in optimization of hyperparameters
    lr_gpr   =0.1       #learning rate for the optimization of the hyperparameters
    convPlot_gpr=True   #plot convergence of optimization of the GPR hyperparameters
    #options for Gauss quadrature test nodes
    nGQtest=[18,18]     #number of test samples in each param dimension
    nMC=100            #number of samples drawn from GPR surrogate to construct estimates 
                        # for the moments of f(q)
    #---------------------------------------------------------
    p=len(distType)  
    #(1) generate synthetic training data
    qTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,trainSampleType,n,qInfo,fExName,noiseType)
    #(2) probabilistic PCE
    ppceDict={'nGQtest':nGQtest,'qInfo':qInfo,'distType':distType,'nIter_gpr':nIter_gpr,
              'lr_gpr':lr_gpr,'convPlot_gpr':convPlot_gpr,'nMC':nMC}
    ppce_=ppce(qTrain,yTrain,noiseSdev,ppceDict)
    optOut=ppce_.optOut
    fMean_samples=ppce_.fMean_samps
    fVar_samples=ppce_.fVar_samps
    #(3) estimate reference mean and varaiance of f(q) using Monte-Carlo approach
    nMC2=100000 
    qMC=[]
    for i in range(p):
        if distType[i]=='Unif':
           sampleType_='unifRand' 
        elif distType[i]=='Norm':
           sampleType='normRand' 
        samps=sampling.trainSample(sampleType=sampleType_,GQdistType=distType[i],
                qInfo=qInfo[i],nSamp=nMC2)
        qMC.append(samps.q)
    fVal_mc=analyticTestFuncs.fEx2D(qMC[0],qMC[1],fExName,'comp').val  
    fMean_mc=np.mean(fVal_mc)
    fVar_mc=np.mean(fVal_mc**2.)-fMean_mc**2.
    #(4) postprocess
    #   (a) plot the exact and GPR response surfaces
    gpr_torch.gprPlot().torch2d_3dSurf(qTrain,yTrain,optOut['qTest'],optOut['post_obs'])
    #   (b) plot histogram and fitted pdf of the mean and variance distributions
    statsUQit.pdfFit_uniVar(fMean_samples,True,[])
    statsUQit.pdfFit_uniVar(fVar_samples,True,[])
    #   (c) compare the reference moments with the estimated values by ppce
    fMean_mean=fMean_samples.mean()
    fMean_sdev=fMean_samples.std()
    fVar_mean=fVar_samples.mean()
    fVar_sdev=fVar_samples.std()
    print(writeUQ.printRepeated('-', 80))
    print('Reference mean(f) = %g' %fMean_mc)
    print('PPCE estimated: E[mean(f)] = %g , sdev[mean(f)] = %g' %(fMean_mean,fMean_sdev))
    print('Reference var(f) = %g' %fVar_mc)
    print('PPCE estimated: E[var(f)] = %g , sdev[var(f)] = %g' %(fVar_mean,fVar_sdev))
#
