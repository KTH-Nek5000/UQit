###############################################################
# Probabilistic Sobol Sensitivity Indices
###############################################################
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
# ToDo: For p>=3, choosing large number of test points to evaluate integrals in Sobol indices, 
#       is problematic since associated samples are generated from GPR (GpyTorch has memory issues). 
#       A solution will be evaluating the integrals for Sobol indices by a quadrature rule. Since
#       we have a complete freedom, using GLL rule would be good. 
import os
import sys
import numpy as np
##import UQit.pce as pce
import UQit.gpr_torch as gpr_torch
import UQit.sobol as sobol
import UQit.stats as statsUQit
import UQit.reshaper as reshaper
##import UQit.sampling as sampling
#
#
class pSobol:
    """
    Probabilistic Sobol (pSobol) Sensitivity Indices over a p-D parameter space, 
    where p=2,3,...
    Model: y=f(q)+e

    Args:
      `qTrain`: 2D numpy array of shape (n,p)
         Training samples for q
      `yTrain`: 1D numpy array of size n
         Training observed values for y
      `noiseV`: 1D numpy array of size n
         Standard-deviation of the noise in the training observations: e~N(0,noiseSdev)
      `pSobolDict`: dict
         Dictionary containing controllers for pSobol, including:
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
      `fMean_list`: 1D numpy array of size `nMC`
         PCE estimates for the mean of f(q)
      `fVar_list`: 1D numpy array of size `nMC`
         PCE estimates for the variance of f(q)
      `optOut`: dict
         Optional outputs for plotting using gprPlot, with the following keys:    
            * `post_f`: Posterior density of f(q)
            * `post_obs`: Posterior density of y
            * `qTest`: A List of length p, 
              =[qTest_1,qTest_2,...,qTest_p], where qTest_i is a 1D numpy array of size 
              `ppceDict['nGQtest'][i]` containing the GQ test samples in the i-th direction
              of the parameter.
    """
    def __init__(self,qTrain,yTrain,noiseV,pSobolDict):
        self.qTrain=qTrain
        self.yTrain=yTrain
        self.noiseV=noiseV
        self.pSobolDict=pSobolDict
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
            if key_ not in self.pSobolDict.keys():
               raise KeyError("%s is required in pSobolDict." %key_) 

        nTest=[]
        for i in range(self.p):
            nTest.append(self.pSobolDict['qTest'][i].shape[0])            
        self.nTest=nTest 

    def cnstrct(self):
        self.pSobol_cnstrct() 

    def pSobol_cnstrct(self):
       """
       Constructing probabilistic Sobol indices over a p-D parameter space, p>1
       """
       p=self.p
       print('... Probabilistic Sobol indices for %d-D input parameter.' %p)
       pSobolDict=self.pSobolDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
#       nGQ=ppceDict['nGQtest']       
       qTest=pSobolDict['qTest'] 
       pdf=pSobolDict['pdf'] 
       nMC=pSobolDict['nMC']           
       nw_=int(nMC/10)
#       distType=ppceDict['distType']
       #Make a dict for gpr (do NOT change)
       gprOpts={'nIter':pSobolDict['nIter_gpr'],    
                'lr':pSobolDict['lr_gpr'],          
                'convPlot':pSobolDict['convPlot_gpr']  
              }
       standardizeYTrain_=False
       if 'standardizeYTrain_gpr' in pSobolDict.keys():
           gprOpts.update({'standardizeYTrain':pSobolDict['standardizeYTrain_gpr']}) 
           standardizeYTrain_=True

#       #Make a dict for PCE (do NOT change)
#       #Always use TP truncation with GQ sampling (hence Projection method) 
#       pceDict={'p':p,
#                'truncMethod':'TP',  
#                'sampleType':'GQ', 
#                'pceSolveMethod':'Projection',
#                'distType':distType
#               }

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
              print("...... pSobol repetition for finding samples of the Sobol indices, iter = %d/%d" 
                      %(j+1,nMC))
       #reshape lists
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
#       #Optional outputs: only used for gprPlot
       optOut={'post_f':post_f,'post_obs':post_obs}
       self.optOut=optOut
#


#########
##MAIN
#########
import UQit.analyticTestFuncs as analyticTestFuncs
import matplotlib.pyplot as plt
import UQit.sampling as sampling
def pSobol_2d_test():
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
          sd=0.5   # noise standard deviation  (Note: non-zero, to avoid instabilities)
          sdV=sd*np.ones(n)
       elif noiseType=='hetero':
          sdV=0.5*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV
    #
    #----- SETTINGS
    #definition of the parameters
    qBound=[[-2,2],[-2,2]]   #Admissible range of parameters

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
    nMC=100           #number of MC samples to compute pSobol indices
    nTest=[41,40]     #number of test points in each parameter dimension to compute integrals in Sobol indices

    #options for GPR
    nIter_gpr_=100        #number of iterations in optimization of GPR hyperparameters
    lr_gpr_   =0.05        #learning rate in the optimization of the hyperparameters
    convPlot_gpr_=True     #plot convergence of optimization of GPR hyperparameters
    #------------------------------------------------
    #(1) Generate training data
    p=len(qBound)    #dimension of the input
    xTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,sampleType,n,qBound,fExName,noiseType)
    print(yTrain.shape)
    nSamp=yTrain.shape[0]
    plot_trainData(nSamp,yTrain_noiseFree,noiseSdev,yTrain)

    #(2) Create the test samples and PDF
    xTestList=[]
    pdf=[]
    for i in range(p):
        #grid_=torch.linspace(qBound[i][0],qBound[i][1],20)    #torch
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        pdf.append(np.ones(nTest[i])/(qBound[i][1]-qBound[i][0]))        
        xTestList.append(grid_)

    #(3) Assemble the pSobolOpts dict
    pSobolDict={'nMC':nMC,'qTest':xTestList,'pdf':pdf,
                'nIter_gpr':nIter_gpr_,'lr_gpr':lr_gpr_,'convPlot_gpr':convPlot_gpr_}

    # (4) Construct probabilistic Sobol indices
    pSobol_=pSobol(xTrain,yTrain,noiseSdev,pSobolDict)
    Si_samps=pSobol_.Si_samps
    Sij_samps=pSobol_.Sij_samps
    STi_samps=pSobol_.STi_samps
    SijName=pSobol_.SijName

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
    plt.xlabel('sample',fontsize=14)
    plt.ylabel('Sobol indices',fontsize=14)
    plt.show()
#
from math import pi
def pSobol_Ishigami_test():
    """
      Test for pSobol() when we have 3 uncertain parameters q1, q2, q3.
      Sobol indices are computed for f(q1,q2,q3)=Ishigami that is analyticTestFuncs.fEx3D('Ishigami').
      The resulting pSobol indices can be compared to standard Sobol indices to verify the implementation of pSobol.
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
    nMC=100           #number of MC samples to compute pSobol indices
    nTest=[20,21,22]     #number of test points in each parameter dimension to compute integrals in Sobol indices
    #options for GPR
    nIter_gpr_=100        #number of iterations in optimization of GPR hyperparameters
    lr_gpr_   =0.05        #learning rate in the optimization of the hyperparameters
    convPlot_gpr_=True     #plot convergence of optimization of GPR hyperparameters
    #--------------------------
    p=len(qBound)
    #(1) Generate training data
    qTrain=sampling.LHS_sampling(n,qBound)  #LHS random samples
    print(qTrain.shape)
    fEx_=analyticTestFuncs.fEx3D(qTrain[:,0],qTrain[:,1],qTrain[:,2],'Ishigami','comp',{'a':a,'b':b})
    print(fEx_.val.shape)
    yTrain=fEx_.val+noise_sdev*np.random.randn(n)
    noiseSdev=noise_sdev*np.ones(n)
#    fEx=np.reshape(fEx_.val,n,'F')

    #(2) Create the test samples and associated PDF
    qTest=[]
    pdf=[]
    for i in range(p):
        qTest.append(np.linspace(qBound[i][0],qBound[i][1],nTest[i]))
        pdf.append(np.ones(nTest[i])/(qBound[i][1]-qBound[i][0]))


    #(3) Assemble the pSobolOpts dict
    pSobolDict={'nMC':nMC,'qTest':qTest,'pdf':pdf,
                'nIter_gpr':nIter_gpr_,'lr_gpr':lr_gpr_,'convPlot_gpr':convPlot_gpr_}

    # (4) Construct probabilistic Sobol indices
    pSobol_=pSobol(qTrain,yTrain,noiseSdev,pSobolDict)
    Si_samps=pSobol_.Si_samps
    Sij_samps=pSobol_.Sij_samps
    STi_samps=pSobol_.STi_samps
    SijName=pSobol_.SijName

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
    plt.xlabel('sample',fontsize=14)
    plt.ylabel('Sobol indices',fontsize=14)
    plt.show()
    

