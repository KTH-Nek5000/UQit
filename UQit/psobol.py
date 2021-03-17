###############################################################
# Probabilistic Sobol Sensitivity Indices
###############################################################
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
# ToDo: For p>=3, choosing large number of test points to evaluate the 
#  integrals in the Sobol indices can be a bit problematic since the 
#  samples generated from GPR (GpyTorch) may lead to memory/CPU issues. 
#  A solution  will be evaluating the integrals for Sobol indices by a 
#  quadrature rule. Since we have a complete freedom, using GLL rule would be good. 
#--------------------------------------------------------------
#
import os
import sys
import numpy as np
import UQit.gpr_torch as gpr_torch
import UQit.sobol as sobol
import UQit.stats as statsUQit
import UQit.reshaper as reshaper
##import UQit.sampling as sampling
#
#
class psobol:
    """
    Probabilistic Sobol (psobol) Sensitivity Indices over a p-D parameter space, 
    where p=2,3,...
    Surrogate model: y=f(q)+e

    Args:
      `qTrain`: 2D numpy array of shape (n,p)
         Training samples for q
      `yTrain`: 1D numpy array of size n
         Training observed values for y
      `noiseV`: 1D numpy array of size n
         Standard-deviation of the noise in the training observations: e~N(0,noiseSdev)
      `psobolDict`: dict
         Dictionary containing controllers for psobol, including:
           * 'qTest': List of length p
              =[qTest_1,...,qTest_p], where qTest_i is a 1d numpy array of size nTest_i containing 
              the uniformly-spaced test points in the i-th dimension
           * 'pdf': List of length p
              =[pdf_1,...,pdf_p], where pdf_i is a 1d numpy array of size nTest_i containing 
              the PDF of q_i evaluated at qTest_i. 
           * 'nMC': int
              Number of independent samples drawn from the GPR to construct PCE 
           * 'nIter_gpr': int
              Number of iterations for optimization of the GPR hyper-parameters
           * 'lr_gpr': float
              Learning rate for optimization of the GPR hyper-parameters
           * 'standardizeYTrain_gpr': bool (optional, default: False)
              If true, the training data are standardized by shifting by mean and scaling by sdev:

              :math:`yStnd =  (yTrain-mean(yTrain))/sdev(yTrain)`  

           * 'convPlot_gpr': bool
              If true, values of the hyper-parameters are plotted vs. iteration during the optimization.
              
    Attributes:
      `Si_samps`: list of length p
         =[S1_samps,S2_samps,...,Sp_samps] where Si_samps is a 1d numpy array of size nMC containing 
         samples of the main Sobol index of the i-th parameter. 
      `Sij_samps`: list of length (p-1)!
         =[S12_samps,S13_samps,...] where Sij_samps is a 1d numpy array of size nMC containing samples 
         of the dual interaction Sobol index between i-th and j-th parameters.
      `STi_samps`: list of length (p-1)!
         =[ST1_samps,ST2_samps,...,STp_samps] where STi_samps is a 1d numpy array of size nMC 
         containing samples of the total Sobol index of the i-th parameter. 
      `SijName`: list of length (p-1)!
         Containing name of the dual-interaction indices.
      `optOut`: dict
         Optional outputs for plotting using gprPlot, with the following keys:    
            * `post_f`: Posterior density of f(q)
            * `post_obs`: Posterior density of y
    """
    def __init__(self,qTrain,yTrain,noiseV,psobolDict):
        self.qTrain=qTrain
        self.yTrain=yTrain
        self.noiseV=noiseV
        self.psobolDict=psobolDict
        self.info()
        self.cnstrct()

    def info(self):
        if self.qTrain.ndim==1:
           raise ValueError("qTrain must be a 2D numpy array.")
        elif self.qTrain.ndim==2:
           self.p=self.qTrain.shape[-1] 

        if self.qTrain.shape[0]==self.yTrain.shape:
           raise ValueError("Size of qTrain and yTrain should be the same.")

        obligKeys=['qTest','pdf','nMC','nIter_gpr','lr_gpr','convPlot_gpr']
        for key_ in obligKeys:
            if key_ not in self.psobolDict.keys():
               raise KeyError("%s is required in psobolDict." %key_) 

        nTest=[]
        for i in range(self.p):
            nTest.append(self.psobolDict['qTest'][i].shape[0])            
        self.nTest=nTest 

    def cnstrct(self):
        self.psobol_cnstrct() 

    def psobol_cnstrct(self):
       """
       Constructing probabilistic Sobol indices over a p-D parameter space, p>1
       """
       p=self.p
       print('... Probabilistic Sobol indices for %d-D input parameter.' %p)
       psobolDict=self.psobolDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
       qTest=psobolDict['qTest'] 
       pdf=psobolDict['pdf'] 
       nMC=psobolDict['nMC']           
       nw_=int(nMC/10)
       #Make a dict for gpr (do NOT change)
       gprOpts={'nIter':psobolDict['nIter_gpr'],    
                'lr':psobolDict['lr_gpr'],          
                'convPlot':psobolDict['convPlot_gpr']  
              }
       standardizeYTrain_=False
       if 'standardizeYTrain_gpr' in psobolDict.keys():
           gprOpts.update({'standardizeYTrain':psobolDict['standardizeYTrain_gpr']}) 
           standardizeYTrain_=True

       #(1) Generate a tensor product grid from qTest. At the grid samples, the gpr is sampled.
       qTestGrid=reshaper.vecs2grid(qTest)

       #(2) Construct GPR surrogate based on training data
       gpr_=gpr_torch.gpr(qTrain,yTrain[:,None],noiseSdev,qTestGrid,gprOpts)
       post_f=gpr_.post_f
       post_obs=gpr_.post_y
       shift_=0.0
       scale_=1.0
       if standardizeYTrain_:
          shift_=gpr_.shift[0]  #0: single-response
          scale_=gpr_.scale[0]    
          
       #optional: plot constructed response surface only for p==2
       #gpr_torch.gprPlot().torch2d_3dSurf(qTrain,yTrain,qTest,post_obs,shift=shift_,scale=scale_)

       #(3) Compute Sobol indices for samples of GPR generated at qTestGrid
       Si_list_=[]
       Sij_list_=[]
       STi_list_=[]
       for j in range(nMC):
           # Draw a sample for f(q) from GPR surrogate
           f_=post_obs.sample().numpy()*scale_+shift_
           f_=np.reshape(f_,self.nTest,'F')
           # Compute the Sobol indices
           sobol_=sobol(qTest,f_,pdf)
           Si_list_.append(sobol_.Si)
           Sij_list_.append(sobol_.Sij)
           STi_list_.append(sobol_.STi)
           if ((j+1)%nw_==0):
              print("...... psobol repetition for finding samples of the Sobol indices, iter = %d/%d" 
                      %(j+1,nMC))
       #reshape lists and arrays
       S_=np.zeros(nMC)
       ST_=np.zeros(nMC)
       Si_list=[] 
       Sij_list=[]
       STi_list=[] 
       for i in range(p):
           for j in range(nMC):
               S_[j]=Si_list_[j][i]
               ST_[j]=STi_list_[j][i]
           Si_list.append(S_.copy())    
           STi_list.append(ST_.copy())    

       for i in range(len(Sij_list_[0])):
           for j in range(nMC):
               S_[j]=Sij_list_[j][i]
           Sij_list.append(S_.copy())    

       self.Si_samps=Si_list
       self.Sij_samps=Sij_list
       self.STi_samps=STi_list
       self.SijName=sobol_.SijName

       #(4) Outputs
       #Optional outputs: can only be used for gprPlot
       optOut={'post_f':post_f,'post_obs':post_obs}
       self.optOut=optOut
#
#
#########
##MAIN
#########
import math as mt
import copy
import matplotlib.pyplot as plt
import UQit.analyticTestFuncs as analyticTestFuncs
import UQit.sampling as sampling
def psobol_2d_test():
    """
    Test for psobol for 2 parameters
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
              grid_=np.linspace(qBound[i][0],qBound[i][1],n[i])
              gridList.append(grid_)
          xTrain=reshaper.vecs2grid(gridList)
        elif sampleType=='random': 
             nSamp=n     #number of random samples   
             xTrain=sampling.LHS_sampling(n,qBound)
        #  (b) Observation noise   
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
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
          sd=0.5   # noise standard deviation  (Note: non-zero, to avoid instabilities)
          sdV=sd*np.ones(n)
       elif noiseType=='hetero':
          sdV=0.5*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV
    #
    #----- SETTINGS
    #definition of the parameters
    qInfo=[[-3,2],[0.5,0.7]]    #information about the parameters
    distType=['Unif','Norm']    #type of distribution of the parameters

    #options for generating training samples
    fExName='type1'          #Type of simulator in analyticTestFuncs.fEx2D
                             #'type1', 'type2', 'type3', 'Rosenbrock'
    sampleType='random'      #'random' or 'grid': type of training samples
    if sampleType=='grid':
       n=[9,9]               #number of training samples in each input dimension
    elif sampleType=='random':
       n=100                 #total number of training samples drawn randomly
    noiseType='hetero'       #noise type: 'homo'=homoscedastic, 'hetero'=heterscedastic

    #options for Sobol indices
    nMC=1000         #number of MC samples to compute psobol indices
    nTest=[41,40]    #number of test points in each parameter dimension to compute integrals in Sobol indices

    #options for GPR
    nIter_gpr_=500        #number of iterations in optimization of GPR hyperparameters
    lr_gpr_   =0.05       #learning rate in the optimization of the hyperparameters
    convPlot_gpr_=True    #plot convergence of optimization of GPR hyperparameters
    #------------------------------------------------
    qBound=copy.deepcopy(qInfo)  #default    
    #(1) Generate training data
    p=len(qInfo)    #dimension of the input
    for i in range(p):
       if distType[i]=='Norm':
          qBound[i][0]=qInfo[i][0]-5.*qInfo[i][1]
          qBound[i][1]=qInfo[i][0]+5.*qInfo[i][1]
    xTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,sampleType,n,qBound,fExName,noiseType)
    nSamp=yTrain.shape[0]
    plot_trainData(nSamp,yTrain_noiseFree,noiseSdev,yTrain)

    #(2) Create the test samples and PDF
    qTest=[]
    pdf=[]
    for i in range(p):
        q_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        if distType[i]=='Norm':
           pdf.append(np.exp(-(q_-qInfo[i][0])**2/(2*qInfo[i][1]**2))/(qInfo[i][1]*mt.sqrt(2*mt.pi)))
        elif distType[i]=='Unif':
           pdf.append(np.ones(nTest[i])/(qBound[i][1]-qBound[i][0]))        
        else:
           raise ValueError("distType of the %d-th parameter can be 'Unif' or 'Norm'"%(i+1))  
        qTest.append(q_)
    #plot PDFs
    for i in range(p):
        plt.plot(qTest[i],pdf[i],label='pdf of q'+str(i+1))
    plt.legend(loc='best')
    plt.show()        

    #(3) Assemble the psobolOpts dict
    psobolDict={'nMC':nMC,'qTest':qTest,'pdf':pdf,
                'nIter_gpr':nIter_gpr_,'lr_gpr':lr_gpr_,'convPlot_gpr':convPlot_gpr_}

    # (4) Construct probabilistic Sobol indices
    psobol_=psobol(xTrain,yTrain,noiseSdev,psobolDict)
    Si_samps=psobol_.Si_samps
    Sij_samps=psobol_.Sij_samps
    STi_samps=psobol_.STi_samps
    SijName=psobol_.SijName

    # Results
    # (a) Compute mean and sdev of the estimated indices
    for i in range(p):
        print('Main Sobol index S%d: mean=%g, sdev=%g',str(i+1),np.mean(Si_samps[i]),np.std(Si_samps[i]))
        print('Total Sobol index ST%d: mean=%g, sdev=%g',str(i+1),np.mean(STi_samps[i]),np.std(STi_samps[i]))
    print('Interacting Sobol indices S12: mean=%g, sdev=%g',np.mean(Sij_samps[0]),np.std(Sij_samps[0]))

    #  (b) plot histogram and fitted pdf to different Sobol indices
    statsUQit.pdfFit_uniVar(Si_samps[0],True,[])
    statsUQit.pdfFit_uniVar(Si_samps[1],True,[])
    statsUQit.pdfFit_uniVar(Sij_samps[0],True,[])

    # (c) plot samples of the Sobol indices
    plt.figure(figsize=(10,5))
    for i in range(p):
        plt.plot(Si_samps[i],'-',label='S'+str(i+1))
        plt.plot(STi_samps[i],'--',label='ST'+str(i+1))
    plt.plot(Si_samps[i],':',label='S'+str(12))
    plt.legend(loc='best',fontsize=14)
    plt.xlabel('Sample',fontsize=14)
    plt.ylabel('Sobol indices',fontsize=14)
    plt.show()
#
from math import pi
def psobol_Ishigami_test():
    """
      Test for psobol for 3 uncertain parameters. 
      Sobol indices are computed for f(q1,q2,q3)=Ishigami available as analyticTestFuncs.fEx3D('Ishigami').
      The resulting psobol indices can be compared to the standard Sobol indices in order to verify 
      the implementation of psobol.
    """
    #--------------------------
    #------- SETTINGS
    #definition of the parameters
    qBound=[[-pi,pi],      #admissible range of parameters
            [-pi,pi],
            [-pi,pi]]
    #options for Training data
#    n=[100, 70, 80]       #number of samples for q1, q2, q3
    n=500 #number of LHS random samples
    a=7   #parameters in Ishigami function
    b=0.1
    noise_sdev=0.2   #standard-deviation of observation noise 
    #options for Sobol indices
    nMC=500           #number of MC samples to compute psobol indices
    nTest=[20,21,22]     #number of test points in each parameter dimension to compute integrals in Sobol indices
    #options for GPR
    nIter_gpr_=100        #number of iterations in optimization of GPR hyperparameters
    lr_gpr_   =0.05        #learning rate in the optimization of the hyperparameters
    convPlot_gpr_=True     #plot convergence of optimization of GPR hyperparameters
    #--------------------------
    p=len(qBound)
    #(1) Generate training data
    qTrain=sampling.LHS_sampling(n,qBound)  #LHS random samples
    fEx_=analyticTestFuncs.fEx3D(qTrain[:,0],qTrain[:,1],qTrain[:,2],'Ishigami','comp',{'a':a,'b':b})
    yTrain=fEx_.val+noise_sdev*np.random.randn(n)
    noiseSdev=noise_sdev*np.ones(n)

    #(2) Create the test samples and associated PDF
    qTest=[]
    pdf=[]
    #qBound=[[-1,1],[-1,1],[-1,1]]
    #for i in range(3):
    #    qTrain[:,i]=(qTrain[:,i]+pi)/(2*pi)*2-1
    for i in range(p):
        qTest.append(np.linspace(qBound[i][0],qBound[i][1],nTest[i]))
        pdf.append(np.ones(nTest[i])/(qBound[i][1]-qBound[i][0]))

    #(3) Assemble the psobolOpts dict
    psobolDict={'nMC':nMC,'qTest':qTest,'pdf':pdf,
                'nIter_gpr':nIter_gpr_,'lr_gpr':lr_gpr_,'convPlot_gpr':convPlot_gpr_}

    # (4) Construct probabilistic Sobol indices
    psobol_=psobol(qTrain,yTrain,noiseSdev,psobolDict)
    Si_samps=psobol_.Si_samps
    Sij_samps=psobol_.Sij_samps
    STi_samps=psobol_.STi_samps
    SijName=psobol_.SijName

    # (a) Compute mean and sdev of the estimated indices
    for i in range(p):
        print('Main Sobol index S%d: mean=%g, sdev=%g',str(i+1),np.mean(Si_samps[i]),np.std(Si_samps[i]))
        print('Total Sobol index ST%d: mean=%g, sdev=%g',str(i+1),np.mean(STi_samps[i]),np.std(STi_samps[i]))
    print('Interacting Sobol indices S12: mean=%g, sdev=%g',np.mean(Sij_samps[0]),np.std(Sij_samps[0]))
    print('Interacting Sobol indices S13: mean=%g, sdev=%g',np.mean(Sij_samps[1]),np.std(Sij_samps[1]))
    print('Interacting Sobol indices S23: mean=%g, sdev=%g',np.mean(Sij_samps[2]),np.std(Sij_samps[2]))

    #  (b) plot histogram and fitted pdf to different Sobol indices
    statsUQit.pdfFit_uniVar(Si_samps[0],True,[])
    statsUQit.pdfFit_uniVar(Si_samps[1],True,[])
    statsUQit.pdfFit_uniVar(Sij_samps[0],True,[])

    # (c) plot samples of the Sobol indices
    plt.figure(figsize=(10,5))
    for i in range(p):
        plt.plot(Si_samps[i],'-',label='S'+str(i+1))
        plt.plot(STi_samps[i],'--',label='ST'+str(i+1))
    plt.plot(Si_samps[i],':',label='S'+str(12))
    plt.legend(loc='best',fontsize=14)
    plt.xlabel('sample',fontsize=14)
    plt.ylabel('Sobol indices',fontsize=14)
    plt.show()
#    
