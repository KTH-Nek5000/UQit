###############################################################
# Probabilistic generalized Polynomial Chaos Expansion (PPCE)
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
import UQit.pce as pce
import UQit.gpr_torch as gpr_torch
import UQit.stats as statsUQit
import UQit.reshaper as reshaper
import UQit.sampling as sampling
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
           * 'qInfo': List of length p
              =[qInfo_1,...,qInfo_p], where qInfo_i is the information about distribution of q_i
                - if q_i~'Unif', qInfo_i =[min(q_i),max(q_i)]
                - if q_i~'Norm', qInfo_i =[m,v] for q~N(m,v^2)
           * 'nMC': int
              Number of independent samples drawn from the GPR to construct PCE 
           * 'nIter_gpr': int
              Number of iterations for optimization of the GPR hyper-parameters
           * 'lr_gpr': float
              Learning rate for optimization of the GPR hyper-parameters
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
       Constructing a probabilistic PCE over a 1D parameter space
       """
       print('... Probabilistic PCE for 1D input parameter.')
       p=self.p
       ppceDict=self.ppceDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
       nGQ=ppceDict['nGQtest']       
       qInfo=ppceDict['qInfo'] 
       nMC=ppceDict['nMC']       
       nw_=int(nMC/10)
       distType=ppceDict['distType']
       #Make a dict for GPR
       gprOpts={'nIter':ppceDict['nIter_gpr'],    
                'lr':ppceDict['lr_gpr'],          
                'convPlot':ppceDict['convPlot_gpr'] 
               }
       #(1) Generate test points that are Gauss quadratures chosen based on the 
       # distribution of q (gPCE rule) 
       sampsGQ=sampling.trainSample(sampleType='GQ',GQdistType=distType,qInfo=qInfo,nSamp=nGQ)
       qTest=sampsGQ.q

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
           pce_=pce.pce(fVal=f_,xi=[],pceDict=pceDict,verbose=False)  #GP+TP
           fMean_list.append(pce_.fMean)
           fVar_list.append(pce_.fVar)
           if ((j+1)%nw_==0):
              print("...... ppce repetition for finding samples of the PCE coefficients, iter = %d/%d" 
                      %(j+1,nMC))

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
       Constructing a probabilistic PCE over a p-D parameter space, p>1
       """
       p=self.p
       print('... Probabilistic PCE for %d-D input parameter.' %p)
       ppceDict=self.ppceDict
       qTrain=self.qTrain
       yTrain=self.yTrain
       noiseSdev=self.noiseV
       #(0) Assignments
       nGQ=ppceDict['nGQtest']       
       qInfo=ppceDict['qInfo'] 
       nMC=ppceDict['nMC']           
       nw_=int(nMC/10)
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
           sampsGQ=sampling.trainSample(sampleType='GQ',GQdistType=distType[i],
                   qInfo=qInfo[i],nSamp=nGQ[i])
           qTestList.append(sampsGQ.q)
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
           pce_=pce.pce(fVal=f_,nQList=nGQ,xi=[],pceDict=pceDict,verbose=False)
           fMean_list.append(pce_.fMean)
           fVar_list.append(pce_.fVar)
           if ((j+1)%nw_==0):
              print("...... ppce repetition for finding samples of the PCE coefficients, iter = %d/%d" 
                      %(j+1,nMC))

       #(4) Outputs
       fMean_list=np.asarray(fMean_list)
       fVar_list=np.asarray(fVar_list)
       #Optional outputs: only used for gprPlot
       optOut={'post_f':post_f,'post_obs':post_obs,'qTest':qTestList}
       self.optOut=optOut
       self.fMean_samps=fMean_list
       self.fVar_samps=fVar_list
#
