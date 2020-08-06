################################################################
# Gaussian Process Regression (GPR)
# Using gpytorch library
#################################################################
#----------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------------
"""
   >>> Tricks I learned about GpyTorch by experience:
    1. Number of learning data cannot exceed 128
       If they do, make batches of max size 128.
    2. If sdev of the observation noise is set to zero, it can
       be problematic due to making Cholesky decomposition impossible.
       Instead of zero, use a very small sdev.
    3. In m-d parameter space, we need tode fine a length-scale     
       per parameter in the kernel. I saw if the parameters range
       are too different or too big,  the kernel's length scales
       may not be optimized during the construction of the GPR.
       To rectify this, original parameter space can be mapped to
       hypercube [-1,1]^m, for instance.
"""    
#----------------------------------------------------------------
#
import os
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
     FixedNoiseGaussianLikelihood,
     HeteroskedasticNoise,
     )
myUQtoolboxPATH=os.getenv("myUQtoolboxPATH")
sys.path.append(myUQtoolboxPATH+'analyticFuncs/')
sys.path.append(myUQtoolboxPATH+'general/')
sys.path.append(myUQtoolboxPATH+'stats/')
import analyticTestFuncs
import reshaper
import sampling
#
#//
#Multitask output, 1D input: y=f(x) \in R^m, m>1, x\in R
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

#Singletask output, 1D input: y=f(x) \in R, x\in R
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

#Singletask output, multi-D input: y=f(x) \in R, x\in R^p, p>1
class SingletaskGPModel_mIn(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingletaskGPModel_mIn, self).__init__(train_x, train_y, likelihood)
        num_dims = train_x.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
#            gpytorch.kernels.RBFKernel(ard_num_dims=num_dims)   #different length scales in different dimentions, RBF
            gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=num_dims)   #different length scales in different dimentions, Matern nu
#            gpytorch.kernels.RBFKernel()   #equal length scales in all input dimensions
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#################################
# GP functions
#################################
#
#////////////////////////////////
def gprTorch_1d(xTrain,yTrain,noiseSdev,xTest,gprOpts):
    """
        GPR for one uncertain parameter, and single/multi-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, (Single Task): 1D numpy array of size n, (Multi Task) 2D numpy array of size nxm (m: dimensionality of Y)
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
               gprOpts: GPR options
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """ 
    nResp=len(yTrain)
    if nResp==1:   #single-task (=single-response)
       post_f,post_obs=gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev,xTest,gprOpts)
    else:          #multi-task (=multi-response)
       post_f,post_obs=gprTorch_1d_multiTask(xTrain,yTrain,noiseSdev,xTest,gprOpts)
    return post_f,post_obs
       
    
#////////////////////////////////
def gprTorch_1d_multiTask(xTrain_,yTrain_,noiseSdev_,xTest,gprOpts):
    """ 
        GPR for one uncertain parameter, and multi-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, 2D numpy array of size nxm (m: dimensionality of Y)
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
               gprOpts: GPR options
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    #------ SETTING
    nIter=200   #number of iterations in optimization of GPR hyperparameters
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
    #(1) convert numpy arrays to torch tensors
    #....
    if 0==0:
       xTrain = torch.linspace(0, 3, 30)
       sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 30)
       sem_y2 = 0.75 - (0.75 - 0.05) * torch.linspace(0, 1, 30)
       yTrain = torch.stack([
                torch.sin(xTrain * (2 * mt.pi)) + sem_y1 * torch.randn(xTrain.size()),
                torch.cos(xTrain * (2 * mt.pi)) + sem_y2 * torch.randn(xTrain.size()),], -1)
       yLogVar = torch.stack([(s**2).log() for s in (sem_y1, sem_y2)], -1)
       print('***********torch training data are used')
    #....

    #(2) assign number of tasks=number of outputs
    #  (a) construct the GPR for noise variance
    log_noise_model = MultitaskGPModel(
                      xTrain,
                      yLogVar,
                      MultitaskGaussianLikelihood(num_tasks=numTasks),
                      num_tasks=numTasks,
                    )
    #  (b) initialize the likelihood
    likelihood = _MultitaskGaussianLikelihoodBase(
                 num_tasks=numTasks,
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
    #  (c) initialize the GPR model (prior)
    model = MultitaskGPModel(xTrain, yTrain, likelihood, num_tasks=numTasks, rank=numTasks)

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

#
#////////////////////////////////////////////////////
def gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev,xTest,gprOpts):
    """ 
        GPR for one uncertain parameter, and 1-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, 1D numpy array of size n
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
               gprOpts: GPR options
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    #(0) Assignments
    nIter=gprOpts['nIter']   #number of iterations in optimization of hyperparameters
    lr_  =gprOpts['lr']      #learning rate for the optimizaer of the hyperparameters
    torch.set_printoptions(precision=8)  #to avoid losing accuracy in print after converting to torch
    #(1) convert numpy arrays to torch tensors
    xTrain=torch.from_numpy(xTrain)
    yTrain=torch.from_numpy(yTrain[0])
    yLogVar=torch.from_numpy(np.log(noiseSdev**2.))

    #(TEST) Replacing numpy-training data with the torch data 
    #if 0==1:
    #   xTrainT = torch.linspace(0, 1, 60)
    #   sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 60)
    #   yTrainT =torch.sin(xTrainT * (2 * mt.pi)) + sem_y1 * torch.randn(xTrainT.size())
    #   yLogVarT =(sem_y1**2).log() 
    #   xTrain=xTrainT
    #   yTrain=yTrainT
    #   yLogVar=yLogVarT
    #(2) Construct GPR for noise
    log_noise_model = SingletaskGPModel(
                      xTrain,
                      yLogVar,
                      GaussianLikelihood(),
                    )
    #(3) Construct GPR for f(q)
    #  (a) Likelihood
    likelihood = _GaussianLikelihoodBase(
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
    #  (b) prior GPR model
    model = SingletaskGPModel(xTrain, yTrain, likelihood)
    #(4) Train the model
    model.train()
    likelihood.train()
    #(5) Optimize the model hyperparameters
    optimizer = torch.optim.Adam([  #Adam optimizaer: https://arxiv.org/abs/1412.6980
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                ], lr=lr_)
    #   "Loss" for GPs - mll: marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses=[]
    lengthSc=[]
    for i in range(nIter):
        optimizer.zero_grad()
        output = model(xTrain)
        loss = -mll(output, yTrain, xTrain)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        lengthSc.append(model.covar_module.base_kernel.lengthscale.item())
        if (i+1) % 100 == 0:
            print('...... GPR-hyperparameters Optimization, iter %d/%d - loss: %.3f - lengthsc: %.3f' % (i + 1, nIter, losses[-1],lengthSc[-1]))
    # Plot convergence of hyperparameters optimization
    if gprOpts['convPlot']:
       plt.figure(figsize=(12,3))
       plt.subplot(121)
       plt.plot(losses,'-r')   
       plt.ylabel('Loss',fontsize=16)
       plt.xlabel('Iteration',fontsize=16)
       plt.xticks(fontsize=13)
       plt.yticks(fontsize=13)
       plt.subplot(122)
       plt.plot(lengthSc,'-b')
       plt.ylabel('Lengthscale',fontsize=16)
       plt.xlabel('Iteration',fontsize=16)
       plt.xticks(fontsize=13)
       plt.yticks(fontsize=13)
       plt.show()

    #(6) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()  
    #(7) Evaluate the posteriors at the test points
    with torch.no_grad():
         xTest=torch.from_numpy(xTest)
         post_f = model(xTest)
         post_obs = likelihood(post_f, xTest)
    return post_f,post_obs

#///////////////////////////////////////////////////////
def gprTorch_pd(xTrain,yTrain,noiseSdev,xTest,gprOpts):
    """ 
        GPR for p>1 uncertain parameter, and single/multi-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models

        Inputs:          
               xTrain: Training model input, 2D numpy array of size nxp
               yTrain: Training model output: multitask: 2D numpy array of size nxm (m: dimensionality of Y)
               yTrain: Training model output, singletask: 1D numpy array of size n
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model input, 2D numpy array of size nTestxp
               gprOpts: GPR options
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    nResp=len(yTrain)
    if nResp==1:   #single-task (=single-response)
       post_f,post_obs = gprTorch_pd_singleTask(xTrain,yTrain[0],noiseSdev,xTest,gprOpts)
    else:          #multi-task (=multi-response)
       print('ERROR in gprTorch_pd(): multitask version is not available yet') 
    return post_f,post_obs

#////////////////////////////////////////////////////
def gprTorch_pd_singleTask(xTrain,yTrain,noiseSdev,xTest,gprOpts):
    """ 
        GPR for p>1 uncertain parameter, and 1-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models

        Inputs:          
               xTrain: Training model input, 2D numpy array of size nxp
               yTrain: Training model output, 1D numpy array of size n
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model input, 2D numpy array of size nTestxp
               gprOpts: GPR options
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    #(0) Assignments
    nIter=gprOpts['nIter']   #number of iterations in optimization of hyperparameters
    lr_  =gprOpts['lr']      #learning rate for the optimizaer of the hyperparameters
    torch.set_printoptions(precision=8)  #to avoid losing accuracy in print after converting to torch
    p=xTrain.shape[-1]       #dimension of the parameter
    #(1) convert numpy arrays to torch tensors
    xTrain=torch.from_numpy(xTrain)
    yTrain=torch.from_numpy(yTrain)
    yLogVar=torch.from_numpy(np.log(noiseSdev**2.))

    #(2) Construct the GPR for the noise
    log_noise_model = SingletaskGPModel_mIn(
                      xTrain,
                      yLogVar,
                      GaussianLikelihood(),
                    )
    #(3) Construct GPR for f(q)
    #  (a) Likelihood
    likelihood = _GaussianLikelihoodBase(
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
   # likelihood = GaussianLikelihood(noise=noiseSdev**2.)   #common Gaussian likelihood with no inference for heteroscedastic noise levels

    #  (b) prior GPR model
    model = SingletaskGPModel_mIn(xTrain, yTrain, likelihood)

    #(3) Optimize the hyperparameters
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
                    {'params': model.parameters()},  # Includes Likelihood parameters
#                    {'params': list(model.parameters()) + list(likelihood.parameters())},
                  ], 
                  lr=lr_)   #lr: learning rate
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses=[]
    lengthSc=[[] for _ in range(p)]
    for i in range(nIter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(xTrain)
        # Calc loss and backprop gradients
        loss = -mll(output, yTrain, xTrain)
        loss.backward()
        # optimize
        optimizer.step()
        # info on optimization
        loss_=loss.item()
        lengthSc_=[]
        for j in range(p):
            lengthSc_.append(model.covar_module.base_kernel.lengthscale.squeeze()[j].item())
#           lengthSC_.append(model.covar_module.base_kernel.lengthscale.item())   #if all lengthscales are the same (see the definition of self.covar_module, above)
        if (i+1) % 100 == 0:
           print('...... GPR-hyperparameters Optimization, iter %d/%d - loss: %.3f' %((i + 1), nIter, loss_),end="  ")
           print('lengthscales='+'%.3f '*p %(tuple(lengthSc_)))
        losses.append(loss_)
        for j in range(p):
            lengthSc[j].append(lengthSc_[j])
        #print('lr=',optimizer.param_groups[0]['lr'])
        #print('pars',optimizer.param_groups[0]['params'])
    # Plot convergence of hyperparameters optimization
    if gprOpts['convPlot']:
       plt.figure(figsize=(12,3))
       plt.subplot(121)
       plt.plot(losses,'-r')   
       plt.ylabel('Loss',fontsize=16)
       plt.xlabel('Iteration',fontsize=16)
       plt.xticks(fontsize=13)
       plt.yticks(fontsize=13)
       plt.subplot(122)
       for j in range(p):
           plt.plot(lengthSc[j],'-',label='Lengthscale, param%d'%(j+1))
       plt.ylabel('Lengthscale',fontsize=16)
       plt.xlabel('Iteration',fontsize=16)
       plt.legend(loc='best',fontsize=14)
       plt.xticks(fontsize=13)
       plt.yticks(fontsize=13)
       plt.suptitle('Convergence of GPR hyperparam optimization')
       plt.show()
    #(4) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()
    #(3) Prediction at test inputs
    with torch.no_grad():
         xTest=torch.from_numpy(xTest)
         post_f = model(xTest)
         post_obs = likelihood(post_f, xTest)
    return post_f,post_obs

##############################
# External Functions for Test
##############################
#///////////////////////////////
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
        return x,y,sdV
      
    print("... gprTorch_1d_test()")
    #----- SETTINGS ----------------
    n=30   #number of training data
    xBound=[0.,3]   #range of input
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    nIter_=800      #number of iterations in optimization of hyperparameters
    lr_   =0.1      #learning rate for the optimizaer of the hyperparameters
    convPlot_=True  #plot convergence of optimization of GPR hyperparameters
    #------------------------------------------------
    #(0) Assigning settings to the gprOpts dict
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_}
    #generate training data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
    gprTorch_1d(xTrain,yTrain,noiseSdev,gprOpts)

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
          sd=0.2   #standard deviation (NOTE: cannot be zero)
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
    n=127   #number of training data
    nTest=100   #number of test data
    xBound=[0.,1]   #range of input
    #type of the noise in the data
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    nIter_=800      #number of iterations in optimization of hyperparameters
    lr_   =0.1      #learning rate for the optimizaer of the hyperparameters
    convPlot_=True  #plot convergence of optimization of GPR hyperparameters
    #------------------------------------------------
    #(0) Assigning settings to the gprOpts dict
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_}
    #(1) Generate training and test data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
    xTest = np.linspace(xBound[0]-0.2, xBound[1]+.2, nTest) #if numpy is used for training
    #xTest = torch.linspace(xBound[0], xBound[1], nTest)   #if torch is used for training
     
    #(2) Construct the GPR using the training data
    post_f,post_obs=gprTorch_1d(xTrain,[yTrain],noiseSdev,xTest,gprOpts)

    #(3) Plots
    with torch.no_grad():
         lower_f, upper_f = post_f.confidence_region()
         lower_obs, upper_obs = post_obs.confidence_region()
         plt.figure(figsize=(10,6))
         plt.plot(xTest,fEx(xTest),'--b')
         plt.plot(xTrain, yTrain, 'ok',markersize=4)
         plt.plot(xTest, post_f.mean[:].numpy(), '-r',lw=2)
         plt.plot(xTest, post_obs.mean[:].numpy(), ':m',lw=2)
         plt.fill_between(xTest, lower_f.numpy(), upper_f.numpy(), alpha=0.3)
         plt.fill_between(xTest, lower_obs.numpy(), upper_obs.numpy(), alpha=0.15, color='r')
         plt.legend(['Exact Reponse','Observed Data', 'Mean Model','Mean Posterior Prediction', 'Confidence (f)', 'Confidence (obs)'],loc='best',fontsize=15)
         #NOTE: confidence = 2* sdev, see 
#         https://github.com/cornellius-gp/gpytorch/blob/4a1ba02d2367e4e9dd03eb1ccbfa4707da02dd08/gpytorch/distributions/multivariate_normal.py
         plt.title('Single-Task GP + Heteroscedastic Noise')
         plt.xticks(fontsize=18)
         plt.yticks(fontsize=18)
         plt.xlabel(r'$\mathbf{q}$',fontsize=17)
         plt.ylabel(r'$y$',fontsize=17)
         plt.show()

#//////////////////////////////////
from mpl_toolkits.mplot3d import Axes3D    #for 3d plot
def gprTorch_2d_singleTask_test():
    """
        Test for GPR for 2d input
    """
    ##
    def plot_trainData(n,fSamples,noiseSdev,yTrain):
        """
           Plot noisy data which are used in GPR. 
           NOTE: The GPR is fitted to yTrain data.
        """
        plt.figure(figsize=(10,5))
        x_=np.zeros(n)
        for i in range(n):
            x_[i]=i+1
        for i in range(500):  #only for plottig possible realizations
            noise_=noiseSdev*np.random.randn(n)
            plt.plot(x_,fSamples+noise_,'.',color='steelblue',alpha=0.4,markersize=1)
        plt.errorbar(x_,fSamples,yerr=1.96*abs(noiseSdev),ls='none',capsize=5,ecolor='k',elinewidth=4,label=r'Observations $95\%$ CI')
        plt.plot(x_,fSamples,'o' ,markersize=6,markerfacecolor='lime',markeredgecolor='salmon',label='Mean Observation')
        plt.plot(x_,yTrain ,'xr' ,markersize=6,label='Sample Observation')
        plt.legend(loc='best',fontsize=15)
        plt.ylabel('QoI',fontsize=17)
        plt.xlabel('Simulation Index',fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)    
        plt.title('Training data with associated confidence')
        plt.show()
    ##
    def gpr_3dsurf_plot(xTrain,yTrain,testGrid,post_obs_mean,upper_obs,lower_obs,upper_f,lower_f):
        """
           3D plot of the GPR surface (mean+CI)
        """
        xTestGrid1,xTestGrid2=np.meshgrid(testGrid[0],testGrid[1], sparse=False, indexing='ij')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        mean_surf = ax.plot_surface(xTestGrid1, xTestGrid2, post_obs_mean,cmap='jet', antialiased=True,rstride=1,cstride=1,linewidth=0,alpha=0.4)
        upper_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, upper_obs, linewidth=1,alpha=0.25,color='r')
        lower_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, lower_obs, linewidth=1,alpha=0.25,color='b')
        #upper_surf_f = ax.plot_wireframe(xTestGrid1, xTestGrid2, upper_f, linewidth=1,alpha=0.5,color='r')
        #lower_surf_f = ax.plot_wireframe(xTestGrid1, xTestGrid2, lower_f, linewidth=1,alpha=0.5,color='b')
        plt.plot(xTrain[:,0],xTrain[:,1],yTrain,'ok')
        plt.show()
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
              #grid_=torch.linspace(qBound[i][0],qBound[i][1],n[i])   #torch
              grid_=np.linspace(qBound[i][0],qBound[i][1],n[i])
              gridList.append(grid_)
          xTrain=reshaper.vecs2grid(gridList[0],gridList[1])
#       xTrain = gpytorch.utils.grid.create_data_from_grid(gridList)  #torch
        elif sampleType=='random': 
             nSamp=n     #number of random samples   
             xi_=sampling.LHS_sampling(nSamp,p)
             xTrain=np.zeros((nSamp,p))
             for i in range(p):
                 xTrain[:,i]=(qBound[i][1]-qBound[i][0])*xi_[:,i]+qBound[i][0]
        #  (b) set the sdev of the observation noise   
        #noiseSdev=torch.ones(nTot).mul(0.1)    #torch
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
        #yTrain = torch.sin(mt.pi*xTrain[:,0])*torch.cos(.25*mt.pi*xTrain[:,1])+torch.randn_like(xTrain[:,0]).mul(0.1)   #torch
        #   (c) Training data
        yTrain=analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'pair')
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
          #sdMin=0.01
          #sdMax=0.5
          #sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
          #sdV=0.15*np.ones(n)
          sdV=0.1*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'pair')+0.001)
       return sdV  #vector of standard deviations
    ##
    def gpr_torch_postProc(post_,nTest):
        """
           Convert the outputs of gpr-torch to numpy format suitable for contourplot
        """
        with torch.no_grad():
            post_mean_=post_.mean.detach().numpy()
            post_mean =post_mean_.reshape((nTest[0],nTest[1]),order='F')   #posterior mean
            lower_, upper_ = post_.confidence_region()     #\pm 2*sdev of posterior mean
            lower_=lower_.detach().numpy().reshape((nTest[0],nTest[1]),order='F')
            upper_=upper_.detach().numpy().reshape((nTest[0],nTest[1]),order='F')
            post_sdev = (post_mean-lower_)/2.0   #sdev of the posterior mean of f(q)
        return post_mean,post_sdev,lower_,upper_

    #----- SETTINGS
    qBound=[[-2,2],[-2,2]]   #Admissible range of parameters
    #options for training data
    fExName='type2'          #Name of Exact function to generate synthetic data
                             #This is typ in fEx2D() in ../../analyticFuncs/analyticFuncs.py
    sampleType='grid'        #'random' or 'grid': type of samples
    if sampleType=='grid':
       n=[9,9]               #number of training observations in each input dimension
    elif sampleType=='random':
       n=100                 #total number of training samples drawn randomly
    noiseType='hetero'       #'homo'=homoscedastic, 'hetero'=heterscedastic
    #options for GPR
    nIter_=1000      #number of iterations in optimization of hyperparameters
    lr_   =0.05      #learning rate for the optimizaer of the hyperparameters
    convPlot_=True  #plot convergence of optimization of GPR hyperparameters
    #options for test points
    nTest=[21,20]     #number of test points in each param dimension
    #------------------------------------------------
    #(0) Assigning settings to the gprOpts dict
    gprOpts={'nIter':nIter_,'lr':lr_,'convPlot':convPlot_}

    #(1) Generate training data
    p=len(qBound)    #dimension of the input
    xTrain,yTrain,noiseSdev,yTrain_noiseFree=trainDataGen(p,sampleType,n,qBound,fExName,noiseType)
    nSamp=yTrain.shape[0]
    plot_trainData(nSamp,yTrain_noiseFree,noiseSdev,yTrain)

    #(2) Create test data
    testGrid=[];
    for i in range(p):
        #grid_=torch.linspace(qBound[i][0],qBound[i][1],20)    #torch
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        testGrid.append(grid_)
    xTest=reshaper.vecs2grid(testGrid[0],testGrid[1])

    #(3) construct the GPR based on the training data and make predictions at test inputs
    post_f,post_obs=gprTorch_pd(xTrain,[yTrain],noiseSdev,xTest,gprOpts)

    #(4) Plot 2d contours
    #   (a) Predicted mean and variance at the test grid    
    post_f_mean,post_f_sdev,lower_f,upper_f=gpr_torch_postProc(post_f,nTest)
    post_obs_mean,post_obs_sdev,lower_obs,upper_obs=gpr_torch_postProc(post_obs,nTest)
    #   (b) Plots
    with torch.no_grad():
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(141)        
        fEx_test=analyticTestFuncs.fEx2D(xTest[:,0],xTest[:,1],fExName,'pair')
        CS0=ax.contour(testGrid[0],testGrid[1],fEx_test.reshape((nTest[0],nTest[1]),order='F').T,levels=40)
        ax.clabel(CS0, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Exact $f(q)$')
        ax = fig.add_subplot(142)
        CS1=ax.contour(testGrid[0],testGrid[1],(post_f_mean).T,levels=40)
        ax.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Mean Posterior of $f(q)$')
        ax = fig.add_subplot(143)
        CS2=ax.contour(testGrid[0],testGrid[1],upper_obs.T,levels=40)
        ax.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Upper Confidence for Observations')
        ax = fig.add_subplot(144)
        CS2=ax.contour(testGrid[0],testGrid[1],lower_obs.T,levels=40)
        ax.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Lower Confidence for Observations')
        plt.show()

        #3d plot
        gpr_3dsurf_plot(xTrain,yTrain,testGrid,post_obs_mean,upper_obs,lower_obs,upper_f,lower_f)

