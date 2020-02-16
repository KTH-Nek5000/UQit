#############################################
# Gaussian Process Regression (GPR)
# Using gpytorch library
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
import sys
import numpy as np
import math as mt
from matplotlib import pyplot as plt
import torch
import gpytorch
from gpytorch.likelihoods import (
    _MultitaskGaussianLikelihoodBase,
     MultitaskGaussianLikelihood,
     GaussianLikelihood,
    _GaussianLikelihoodBase,
     HeteroskedasticNoise,
     )

#Multitask GP, y=f(x) \in R^m, m>1
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=0):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

#Singletask GP, y=f(x) \in R^m, m=1
class SingletaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingletaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#////////////////////////////////
def gprTorch_1d(xTrain,yTrain,noiseSdev):
    """ 
        GPR for one uncertain parameter, y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise can have different standard devations. 
        - Supports both homo- and hetero-scedastic models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, mD numpy array of size nxm (m: number of responses)
               noiseSdev: A 1D numpy vecor, standard deviation of the Gaussian noise in each of observations
        Outputs:
    """
    nResp=len(yTrain)
    if nResp>1: #multi-task (multi-response)
       model,likelihood=gprTorch_1d_multiTask(xTrain,yTrain,noiseSdev)
    else:
       model,likelihood=gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev)
    return model,likelihood 
       
    
#////////////////////////////////
def gprTorch_1d_multiTask(xTrain_,yTrain_,noiseSdev_):
    """ 
        GPR for one uncertain parameter, y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise can have different standard devations. 
        - Supports both homo- and hetero-scedastic models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, mD numpy array of size nxm (m: number of responses)
               noiseSdev: A 1D numpy vecor, standard deviation of the Gaussian noise in each of observations
        Outputs:
    """
    #---- SETTING
    nIter=20   #number of iterations in optimization of GPR hyperparameters
    #-------------------------------
    #(0) assignment and reformat inputs consistent with gpytorch
    numTasks=len(yTrain_)
    #numpy -> torch
    yTrain=[]
    noiseSdev=[]
    xTrain=torch.from_numpy(xTrain_)
    for i in range(numTasks):
        yTrain.append(torch.from_numpy(yTrain_[i]))
        noiseSdev.append(torch.from_numpy(noiseSdev_[i]))
    #make torch.stacks
    yTrain=torch.stack([yTrain[0],yTrain[1]],-1)
    yLogVar = torch.stack([(s**2).log() for s in (noiseSdev[0],noiseSdev[1])], -1)
    print('Mine')
    print(yLogVar)
    print(yLogVar.shape)
    #(1) convert numpy arrays to torch tensors

    #....
    xTrain = torch.linspace(0, 3, 30)
    sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 30)
    sem_y2 = 0.75 - (0.75 - 0.05) * torch.linspace(0, 1, 30)
    yTrain = torch.stack([
              torch.sin(xTrain * (2 * mt.pi)) + sem_y1 * torch.randn(xTrain.size()),
              torch.cos(xTrain * (2 * mt.pi)) + sem_y2 * torch.randn(xTrain.size()),], -1)
    yLogVar = torch.stack([(s**2).log() for s in (sem_y1, sem_y2)], -1)
    print('original')
    print(yLogVar)
    print(yLogVar.shape)
    #....



    #(2) assign number of tasks=number of outputs
#    numTasks=1   
#    if yTrain.ndim>1:
#       numTasks=yTrain.shape[1]
#    print('numTasks=',numTasks)
    #(3) construct log of noise variance
#    for i in range(len(noiseSdev)):
#        noiseSdev_=torch.from_numpy(noiseSdev[i])
#        yLogVar=torch.stack([(s**2.).log() for s in (noiseSdev_)],-1)
    #(4) construct the GPR model
    #  (a) construct the GPR for noise variance
    log_noise_model = MultitaskGPModel(
                      xTrain,
                      yLogVar,
                      MultitaskGaussianLikelihood(num_tasks=numTasks),
                      num_tasks=numTasks,
                    )
#    log_noise_model = SingletaskGPModel(
#                      xTrain,
#                      yLogVar,
#                      GaussianLikelihood(),
#                    )
    #  (b) initialize the likelihood
#    if numTasks>1
    likelihood = _MultitaskGaussianLikelihoodBase(
                 num_tasks=numTasks,
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
#    likelihood = MultitaskGaussianLikelihood(num_tasks=numTasks)
#    likelihood = GaussianLikelihood()
#    likelihood = GaussianLikelihood(
#                 noise_covar=HeteroskedasticNoise(log_noise_model),
#               )
    #  (c) initialize the GPR model
#    if numTask>1
    model = MultitaskGPModel(xTrain, yTrain, likelihood, num_tasks=numTasks, rank=numTasks)
#    model = SingletaskGPModel(xTrain, yTrain, likelihood)

    #(4) train the model
    model.train()
    likelihood.train()

    #(5) optimize the model hyperparameters
    optimizer = torch.optim.Adam([  #Adam optimizaer: https://arxiv.org/abs/1412.6980
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                ], lr=0.1)
    #"Loss" for GPs - mll: marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    print('after')
    for i in range(nIter):
        optimizer.zero_grad()
        print('after2')
        output = model(xTrain)
        print('after3',output,xTrain.shape,yTrain.shape)
        loss = -mll(output, yTrain, xTrain)
        print('after4')
        loss.backward()
        if (i+1) % 10 == 0:
           print('...... GPR-hyperParam Optimization, iter %d/%d - loss: %.3f' % (i + 1, nIter, loss.item()))
        optimizer.step()
    print('afterOut')
    #(6) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()  

    return model,likelihood


#////////////////////////////////
def gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev):
    """ 
        GPR for one uncertain parameter, y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise can have different standard devations. 
        - Supports both homo- and hetero-scedastic models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, mD numpy array of size nxm (m: number of responses)
               noiseSdev: A 1D numpy vecor, standard deviation of the Gaussian noise in each of observations
        Outputs:
    """
    #---- SETTING
    nIter=100   #number of iterations in optimization of GPR hyperparameters
    #-------------------------------
    #(1) convert numpy arrays to torch tensors
    xTrain=torch.from_numpy(xTrain)
    yTrain=torch.from_numpy(yTrain[0])
#    yTrain = torch.stack([torch.from_numpy(yTrain[0])],-1)
#    noiseSdev=torch.from_numpy(noiseSdev)
    yLogVar=torch.from_numpy(np.log(noiseSdev**2.))

    #....Replacing numpy-training data with the torch data
    if 0==1:
       xTrainT = torch.linspace(0, 1, 60)
       sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 60)
       yTrainT =torch.sin(xTrainT * (2 * mt.pi)) + sem_y1 * torch.randn(xTrainT.size())
       yLogVarT =(sem_y1**2).log() 
       xTrain=xTrainT
       yTrain=yTrainT
       yLogVar=yLogVarT
    #.....................................................
    #(2) assign number of tasks=number of outputs
    log_noise_model = SingletaskGPModel(
                      xTrain,
                      yLogVar,
                      GaussianLikelihood(),
                    )
    #  (b) initialize the likelihood
    likelihood = _GaussianLikelihoodBase(
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
    #  (c) initialize the GPR model
    model = SingletaskGPModel(xTrain, yTrain, likelihood)

    #(4) train the model
    model.train()
    likelihood.train()
    #(5) optimize the model hyperparameters
    optimizer = torch.optim.Adam([  #Adam optimizaer: https://arxiv.org/abs/1412.6980
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                ], lr=0.1)
    #"Loss" for GPs - mll: marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(nIter):
        optimizer.zero_grad()
        output = model(xTrain)
        loss = -mll(output, yTrain, xTrain)
        loss.backward()
        if (i+1) % 10 == 0:
           print('...... GPR-hyperParam Optimization, iter %d/%d - loss: %.3f' % (i + 1, nIter, loss.item()))
        optimizer.step()

    #(6) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()  
    return model,likelihood




##############################
# External Functions for Test
##############################
#//////////////////////////////////
def gprTorch_1d_multiTask_test():
    """
        Test for GPR for a mult-task model which depends on 1 parameter
    """
    def fEx(x,i):
        """
          Exact model output
        """
        if i==0:
           yEx=np.sin(2*mt.pi*x)
        elif i==1:
           yEx=np.cos(2*mt.pi*x)
        return yEx

    def noiseGen(n,noiseType,i_):
       """
          Generate a 1D numpy array of standard deviations of independent Gaussian noises
       """
       if noiseType=='homo':
          sd =0.1   #standard deviation for model
          sdV=np.asarray([sd]*n)
       elif noiseType=='hetero':
          sdMin=[0.05,0.55]
          sdMax=[0.15,0.25]
          sdV=[]
       sdV_=sdMin[i_]+(sdMax[i_]-sdMin[i_])*np.linspace(0.0,3.0,n)
       return sdV_  #vector of standard deviations

    def trainData(xBound,n,noiseType):
        """
           Create training data D={X,Y}
        """
        x=np.linspace(xBound[0],xBound[1],n)
        y=[]
        sdV=[]
        for i in range(2):   #2 tasks
            #y=fEx(x) + sdV * np.random.randn(n)
            sdV_=noiseGen(n,noiseType,i)
            sdV.append(np.asarray(sdV_))
            y_=np.asarray(fEx(x,i) + sdV_ * np.random.randn(n)) 
            y.append(y_)
        #y=np.asarray(y)
        #sdV=np.asarray(sdV)
        return x,y,sdV
      
    print("... gprTorch_1d_test()")
    #----- SETTINGS ----------------
    n=30   #number of training data
    xBound=[0.,3]   #range of input
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    #-------------------------------
    #generate training data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)

    #
#    gprTorch_1d(xTrain,[yTrain],[noiseSdev])
    gprTorch_1d(xTrain,yTrain,noiseSdev)


    #plots
    #plt.plot(xTrain,yTrain,'ob')
    #xTest=np.linspace(xBound[0],xBound[1],100)
    #plt.plot(xTest,fEx(xTest),'-r')
    #plt.show()
  


#//////////////////////////////////
def gprTorch_1d_singleTask_test():
    """
        Test for GPR for 1d parameter
    """
    def fEx(x):
        """
          Exact model output
        """
        yEx=np.sin(2*mt.pi*x)
        return yEx

    def noiseGen(n,noiseType):
       """
          Generate a 1D numpy array of standard deviations of independent Gaussian noises
       """
       if noiseType=='homo':
          sd=0.25   #standard deviation
          sdV=[sd]*n
          sdV=np.asarray(sdV)
       elif noiseType=='hetero':
          sdMin=0.05
          sdMax=0.55
          sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
       return sdV  #vector of standard deviations

    def trainData(xBound,n,noiseType):
        """
           Create training data D={X,Y}
        """
        x=np.linspace(xBound[0],xBound[1],n)
        sdV=noiseGen(n,noiseType)
        y=fEx(x) + sdV * np.random.randn(n)
        return x,y,sdV
      
    print("... gprTorch_1d_test()")
    #----- SETTINGS ----------------
    n=50   #number of training data
    nTest=200   #number of test data
    xBound=[0.,1]   #range of input
    #type of the noise in the data
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    #-------------------------------
    #generate training data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
     
    #construct the GPR using the training data
    model,likelihood=gprTorch_1d(xTrain,[yTrain],noiseSdev)

    #plots
    with torch.no_grad():
         #test_x = torch.linspace(xBound[0], xBound[1], nTest)   #if torch used for training
         test_x = np.linspace(xBound[0]-0.2, xBound[1]+.2, nTest) #if numpy used for training
         test_x=torch.from_numpy(test_x)
         post_f = model(test_x)
         post_obs = likelihood(post_f, test_x)
    with torch.no_grad():
         lower_f, upper_f = post_f.confidence_region()
         lower_obs, upper_obs = post_obs.confidence_region()
         plt.plot(xTrain, yTrain, 'ok',markersize=4)
         plt.plot(test_x, post_f.mean[:].numpy(), '-b')
         plt.fill_between(test_x.numpy(), lower_f.numpy(), upper_f.numpy(), alpha=0.3)
         plt.fill_between(test_x.numpy(), lower_obs.numpy(), upper_obs.numpy(), alpha=0.25, color='r')
         plt.legend(['Observed Data', 'Mean', 'Confidence (f)', 'Confidence (obs)'])
         plt.title('Single-Task GP + Heteroscedastic Noise')
         plt.show()


#    plt.plot(xTrain,yTrain,'ob')
#    xTest=np.linspace(xBound[0],xBound[1],100)
#    plt.plot(xTest,fEx(xTest),'-r')
#    plt.show()
  
