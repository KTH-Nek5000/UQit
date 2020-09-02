###############################################################
# Probabilistic generalized Polynomial Chaos Expansion (PPCE)
#     PPCE = gPCE + GPR
###############################################################
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
#TODO:
#2. Add other distType for the pce
#--------------------------------------------------------------
#
import os
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
UQit=os.getenv("UQit")
sys.path.append(UQit)
import pce
import gpr_torch
import pdfHisto
import analyticTestFuncs
import writeUQ
import reshaper
import sampling
#
#
class ppce:
    """
    Probabilistic Polynomial Chaos Expansion (PPCE) over a p-D parameter space, 
    where p=1,2,...
    Model: y=f(q)+e
    Args:
      `qTrain`: 2D numpy array of shape (n,p)
         Training samples for q
      `yTrain`: 1D numpy array of size n
         Training observed values for y
      `noiseV`: 1D numpy array of size n
         Standard-deviation of the noise in the training observations: e~N(0,noiseSdev)
      `ppceDict`: dict
         Dictionary containing controllers for PPCE, including:
           * 'nGQtest': List of length p
              Number of GQ test points in each of the p-directions 
           * 'qBound': List of length p
              =[qBound_1,...,qBound_p], where qBound_i is the admissible range of q_i
           * 'nMC': int
              Number of independent samples drawn from the GPR to construct PCE 
           * 'nIter_gpr': int
              Number of iterations for optimization of the GPR hyper-parameters
           * 'lr_gpr': float
              Learning rate for optimization of the GPR hyper-parameters
    Attributes:
       `fMean_list`: 1D numpy array of size `nMC`
          PCE estimates for the mean of f(q)
       `fVar_list`: 1D numpy array of size `nMC`
          PCE estimates for the variance of f(q)
       `optOut`: dict
          Optional outputs for plotting using gprPlot, with the following keys:    
            * `post_f`: Posterior density of f(q)
            * `post_obs`: Posterior density of y
            * `qTest`: A List of length p
              =[qTest_1,qTest_2,...,qTest_p], where qTest_i is a 1D numpy array of size 
               `ppceDict['nGQtest'][i]` containing the GQ test samples in the i-th direction
               of the parameter.
    """
    def __init__(self,qTrain,yTrain,noiseV,ppceDict):
        self.qTrain=qTrain
        self.yTrain=yTrain
        self.noiseV=noiseV
        self.ppceDict=ppceDict
        self.info()
        self.cnstrct()

    def info(self):
        if self.qTrain.ndim==1:
           self.p=1
        elif self.qTrain.ndim==2:
           self.p=self.qTrain.shape[-1] 

        if self.qTrain.shape[0]==self.yTrain.shape:
           raise ValueError("Size of qTrain and yTrain should be the same.")

    def cnstrct(self):
        if self.p==1:
           self.ppce_cnstrct_1d() 
        elif self.p>1:
           self.ppce_cnstrct_pd() 

    def ppce_cnstrct_1d(self):
       """
       Constructing a probabistric PCE over a 1D parameter space
       """
       print('... Probabilistic PCE for 1D input parameter.')
       p=self.p
       ppceDict=self.ppceDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
       nGQ=ppceDict['nGQtest']       
       qBound=ppceDict['qBound'] 
       nMC=ppceDict['nMC']       
       distType=ppceDict['distType']
       #Make a dict for GPR
       gprOpts={'nIter':ppceDict['nIter_gpr'],    
                'lr':ppceDict['lr_gpr'],          
                'convPlot':ppceDict['convPlot_gpr'] 
               }

       #(1) Generate test points that are Gauss quadratures chosen based on the 
       # distribution of q (gPCE rule) 
       xiGQ,wGQ=pce.pce.gqPtsWts(nGQ,distType)  
       if distType=='Unif':
          qTest=pce.pce.mapFromUnit(xiGQ,qBound) 

       #(2) Construct GPR surrogate based on training data
       gpr_=gpr_torch.gpr(qTrain[:,None],yTrain[:,None],noiseSdev,qTest[:,None],gprOpts)
       post_f=gpr_.post_f
       post_obs=gpr_.post_y

       #(3) Use samples of GPR tested at GQ nodes to construct a PCE
       fMean_list=[]      
       fVar_list =[]      
       pceDict={'p':p,'sampleType':'GQ','pceSolveMethod':'Projection','distType':[distType]}
       for j in range(nMC):
           # Draw a sample for f(q) from GPR surrogate
           f_=post_obs.sample().numpy()
           # Construct PCE for the drawn sample
           pce_=pce.pce(fVal=f_,xi=[],pceDict=pceDict)
           fMean_list.append(pce_.fMean)
           fVar_list.append(pce_.fVar)
           if ((j+1)%50==0):
              print("...... ppce repetition for finding samples of the PCE coefficients, iter = %d/%d" 
                      %(j,nMC))

       #(4) Outputs
       fMean_list=np.asarray(fMean_list)
       fVar_list=np.asarray(fVar_list)
       #Optional outputs: only used for gprPlot
       optOut={'post_f':post_f,'post_obs':post_obs,'qTest':[qTest]}
       self.fMean_samps=fMean_list
       self.fVar_samps=fVar_list
       self.optOut=optOut
#
    def ppce_cnstrct_pd(self):
       """
       Constructing a probabistric PCE over a p-D parameter space, p>1
       """
       p=self.p
       print('... Probabilistic PCE for %d-D input parameter.' %p)
       ppceDict=self.ppceDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
       nGQ=ppceDict['nGQtest']       
       qBound=ppceDict['qBound']     
       nMC=ppceDict['nMC']           
       distType=ppceDict['distType']
       #Make a dict for gpr (do NOT change)
       gprOpts={'nIter':ppceDict['nIter_gpr'],    
                'lr':ppceDict['lr_gpr'],          
                'convPlot':ppceDict['convPlot_gpr']  
              }
       #Make a dict for PCE (do NOT change)
       #Always use TP truncation with GQ sampling (hence Projection method) 
       pceDict={'p':p,
                'truncMethod':'TP',  
                'sampleType':'GQ', 
                'pceSolveMethod':'Projection',
                'distType':distType
               }

       #(1) Generate test points that are Gauss quadratures chosen based on 
       # the distribution of q (gPCE rule) 
       qTestList=[]
       for i in range(p):
           xiGQ_,wGQ_=pce.pce.gqPtsWts(nGQ[i],distType[i]) 
           if distType[i]=='Unif':
              qTestList.append(pce.pce.mapFromUnit(xiGQ_,qBound[i])) 
       qTestGrid=reshaper.vecs2grid(qTestList)

       #(2) Construct GPR surrogate based on training data
       gpr_=gpr_torch.gpr(qTrain,yTrain[:,None],noiseSdev,qTestGrid,gprOpts)
       post_f=gpr_.post_f
       post_obs=gpr_.post_y

       #(3) Use samples of GPR tested at GQ nodes to construct a PCE
       fMean_list=[]      
       fVar_list =[]      
       for j in range(nMC):
           # Draw a sample for f(q) from GPR surrogate
           f_=post_obs.sample().numpy()
           # Construct PCE for the drawn sample
           pce_=pce.pce(fVal=f_,nQList=nGQ,xi=[],pceDict=pceDict)
           fMean_list.append(pce_.fMean)
           fVar_list.append(pce_.fVar)
           if ((j+1)%50==0):
              print("...... ppce repetition for finding samples of the PCE coefficients, iter = %d/%d" 
                      %(j,nMC))

       #(4) Outputs
       fMean_list=np.asarray(fMean_list)
       fVar_list=np.asarray(fVar_list)
       #Optional outputs: only used for gprPlot
       optOut={'post_f':post_f,'post_obs':post_obs,'qTest':qTestList}
       self.optOut=optOut
       self.fMean_samps=fMean_list
       self.fVar_samps=fVar_list
#
#
# Tests
#
import torch   #for plot
def ppce_1d_test():
    """
        Test PPCE over 1D parameter space
    """
    def fEx(x):
        """
           Exact simulator
        """
        #yEx=np.sin(2*mt.pi*x)
        yEx=analyticTestFuncs.fEx1D(x,fType,qBound).val
        return yEx
    #
    def noiseGen(n,noiseType):
        """
           Generate a 1D numpy array of standard deviations of independent Gaussian noises
        """
        if noiseType=='homo': #homoscedastic noise 
           sd=0.1   #standard deviation (NOTE: cannot be zero, but can be very small)
           sdV=[sd]*n
           sdV=np.asarray(sdV)
        elif noiseType=='hetero': #heteroscedastic noise
           sdMin=0.02
           sdMax=0.2
           sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
        return sdV  #vector of standard deviations
    #
    def trainData(xBound,n,noiseType):
        """
          Create training data D={X,Y}
        """
        x=np.linspace(xBound[0],xBound[1],n)
        sdV=noiseGen(n,noiseType)
        y=fEx(x) + sdV * np.random.randn(n)
        return x,y,sdV
    #
    #-------SETTINGS------------------------------
    n=12       #number of training data
    nGQtest=50   #number of test points (=Gauss Quadrature points)
    qBound=[0,1]   #range of input
    #type of the noise in the data
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    distType='Unif'
    #GPR options
    nIter_gpr=800      #number of iterations in optimization of hyperparameters
    lr_gpr   =0.1      #learning rate for the optimizaer of the hyperparameters    
    convPlot_gpr=True  #plot convergence of optimization of GPR hyperparameters
    #number of samples drawn from GPR surrogate to construct estimates for moments of f(q)
    nMC=1000
    #---------------------------------------------    
    if distType=='Unif':
       fType='type1' 
    #(1) Generate synthetic training data
    qTrain,yTrain,noiseSdev=trainData(qBound,n,noiseType)
    #(2) Probabilistic gPCE 
    #   (a) make the dictionary
    ppceDict={'nGQtest':nGQtest,'qBound':qBound,'distType':distType,'nIter_gpr':nIter_gpr,'lr_gpr':lr_gpr,'convPlot_gpr':convPlot_gpr,'nMC':nMC}
    #   (b) call the method
    ppce_=ppce(qTrain,yTrain,noiseSdev,ppceDict)
    fMean_samples=ppce_.fMean_samps
    fVar_samples=ppce_.fVar_samps
    optOut=ppce_.optOut

    #(3) postprocess
    #   (a) plot the GPR surrogate along with response from the exact simulator    
    pltOpts={'title':'PPCE, 1d param, %s-scedastic noise'%noiseType}
    gpr_torch.gprPlot(pltOpts).torch1d(optOut['post_f'],optOut['post_obs'],qTrain,yTrain,optOut['qTest'][0],fEx(optOut['qTest'][0]))

    #   (b) plot histogram and pdf of the mean and variance distribution 
    pdfHisto.pdfFit_uniVar(fMean_samples,True,[])
    pdfHisto.pdfFit_uniVar(fVar_samples,True,[])
    #   (c) compare the exact moments with estimated values by ppce
    fEx=analyticTestFuncs.fEx1D(qTrain,fType,qBound)
    fEx.moments(qBound)
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
        Test for ppce_pd_cnstrct()
    """
    ##
    def trainDataGen(p,sampleType,n,qBound,fExName,noiseType):
        """
           Generate Training Data
        """
        #  (a) xTrain
        if sampleType=='grid':
          nSamp=n[0]*n[1]
          gridList=[];
          for i in range(p):
              grid_=np.linspace(qBound[i][0],qBound[i][1],n[i])
              gridList.append(grid_)
          xTrain=reshaper.vecs2grid(gridList)
        elif sampleType=='random':
             nSamp=n
             xTrain=sampling.LHS_sampling(nSamp,qBound)
        #  (b) set the sdev of the observation noise
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
        #  (c) Training data
        yTrain=analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val
        yTrain_noiseFree=yTrain
        yTrain=yTrain_noiseFree+noiseSdev*np.random.randn(nSamp)
        return xTrain,yTrain,noiseSdev,yTrain_noiseFree
    ##
    def noiseGen(n,noiseType,xTrain,fExName):
       """
          Generate a 1D numpy array of standard deviations of independent Gaussian noises
       """
       if noiseType=='homo':
          sd=0.2   #standard deviation (NOTE: cannot be zero)
          sdV=sd*np.ones(n)
       elif noiseType=='hetero':
          sdV=0.1*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV  #vector of standard deviations
    ##
    #
    #----- SETTINGS -------------------------------------------
    qBound=[[-2,2],[-2,2]]   #Admissible range of parameters
    distType=['Unif','Unif']
    #options for training data
    fExName='type2'          #Name of Exact function to generate synthetic data
                             #This is typ in fEx2D() in ../../analyticFuncs/analyticFuncs.py
    trainSampleType='random'        #'random' or 'grid': type of samples
    if trainSampleType=='grid':
       n=[10,10]               #number of training observations in each input dimension
    elif trainSampleType=='random':
       n=100               #total number of training samples drawn randomly
    #NOTE: there might be limitation for n*nGQtest because of torch, since we are not using batch   
    noiseType='homo'       #'homo'=homoscedastic, 'hetero'=heterscedastic
    #options for GPR
    nIter_gpr=1000      #number of iterations in optimization of hyperparameters
    lr_gpr   =0.1      #learning rate for the optimizaer of the hyperparameters
    convPlot_gpr=True  #plot convergence of optimization of GPR hyperparameters
    #options for Gauss Quadrature test points
    nGQtest=[18,18]     #number of test points in each param dimension
    #number of samples drawn from GPR surrogate to construct estimates for moments of f(q)
    nMC=200
    #---------------------------------------------------------
    p=len(distType)  #dimension of the input parameter q
    #(1) Generate synthetic training data    
    qTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,trainSampleType,n,qBound,fExName,noiseType)
    #(2) Probabilistic gPCE 
    #   (a) make the dictionary
    ppceDict={'nGQtest':nGQtest,'qBound':qBound,'distType':distType,'nIter_gpr':nIter_gpr,'lr_gpr':lr_gpr,'convPlot_gpr':convPlot_gpr,'nMC':nMC}
    #   (b) call the method
    ppce_=ppce(qTrain,yTrain,noiseSdev,ppceDict)
    optOut=ppce_.optOut
    fMean_samples=ppce_.fMean_samps
    fVar_samples=ppce_.fVar_samps

    #(3) postprocess
    #   (a) plot the GPR surrogate along with response from the exact simulator    
    gpr_torch.gprPlot().torch2d_3dSurf(qTrain,yTrain,optOut['qTest'],nGQtest,optOut['post_obs'],optOut['post_f'])

    #   (b) plot histogram and pdf of the mean and variance distribution 
    pdfHisto.pdfFit_uniVar(fMean_samples,True,[])
    pdfHisto.pdfFit_uniVar(fVar_samples,True,[])
    #   (c) compare the exact moments with estimated values by ppce
    fMean_mean=fMean_samples.mean()
    fMean_sdev=fMean_samples.std()
    fVar_mean=fVar_samples.mean()
    fVar_sdev=fVar_samples.std()
    print(writeUQ.printRepeated('-', 80))
    #print('>> Exact mean(f) = %g' %fMean_ex)
    print('   ppce estimated: E[mean(f)] = %g , sdev[mean(f)] = %g' %(fMean_mean,fMean_sdev))
    #print('>> Exact Var(f) = %g' %fVar_ex)
    print('   ppce estimated: E[Var(f)] = %g , sdev[Var(f)] = %g' %(fVar_mean,fVar_sdev))


