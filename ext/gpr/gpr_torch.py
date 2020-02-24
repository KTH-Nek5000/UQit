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
     FixedNoiseGaussianLikelihood,
     HeteroskedasticNoise,
     )
sys.path.append('../../analyticFuncs/')
sys.path.append('../../general/')
sys.path.append('../../stats/')
import analyticTestFuncs
import reshaper
import sampling

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

#Singletask output, multi-D input: y=f(x) \in R, x\in R^d, d>1
class SingletaskGPModel_mIn(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingletaskGPModel_mIn, self).__init__(train_x, train_y, likelihood)
        num_dims = train_x.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_dims)   #different length scales in different dimentions
#            gpytorch.kernels.RBFKernel()
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#################################
# general functions
#################################


#################################
# GP functions
#################################

#////////////////////////////////
def gprTorch_1d(xTrain,yTrain,noiseSdev,xTest):
    """
        GPR for one uncertain parameter, and single/multi-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, (Single Task): 1D numpy array of size n, (Multi Task) 2D numpy array of size nxm (m: dimensionality of Y)
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """ 
    nResp=len(yTrain)
    if nResp==1:   #single-task (=single-response)
       post_f,post_obs=gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev,xTest)
    else:          #multi-task (=multi-response)
       post_f,post_obs=gprTorch_1d_multiTask(xTrain,yTrain,noiseSdev,xTest)
    return post_f,post_obs
       
    
#////////////////////////////////
def gprTorch_1d_multiTask(xTrain_,yTrain_,noiseSdev_,xTest):
    """ 
        GPR for one uncertain parameter, and multi-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, 2D numpy array of size nxm (m: dimensionality of Y)
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
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
       print(yLogVar)
       print(yLogVar.shape)
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


#////////////////////////////////////////////////////
def gprTorch_1d_singleTask(xTrain,yTrain,noiseSdev,xTest):
    """ 
        GPR for one uncertain parameter, and 1-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models
        Inputs:          
               xTrain: Training model input, 1D numpy array of size n
               yTrain: Training model output, 1D numpy array of size n
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model inputs, 1d numpy array of size nTest
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    #---------- SETTING
    nIter=200   #number of iterations in optimization of GPR hyperparameters
    #-------------------------------
    #(1) convert numpy arrays to torch tensors
    xTrain=torch.from_numpy(xTrain)
    yTrain=torch.from_numpy(yTrain[0])
    yLogVar=torch.from_numpy(np.log(noiseSdev**2.))

    #(TEST) Replacing numpy-training data with the torch data 
    if 0==1:
       xTrainT = torch.linspace(0, 1, 60)
       sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 60)
       yTrainT =torch.sin(xTrainT * (2 * mt.pi)) + sem_y1 * torch.randn(xTrainT.size())
       yLogVarT =(sem_y1**2).log() 
       xTrain=xTrainT
       yTrain=yTrainT
       yLogVar=yLogVarT
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
                ], lr=0.01)
    #   "Loss" for GPs - mll: marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(nIter):
        optimizer.zero_grad()
        output = model(xTrain)
        loss = -mll(output, yTrain, xTrain)
        loss.backward()
        if (i+1) % 10 == 0:
           print('...... GPR-hyperparameters Optimization, iter %d/%d - loss: %.3f' % (i + 1, nIter, loss.item()))
        optimizer.step()
    #(6) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()  
    #(7) Evaluate the posteriors at the test points
    with torch.no_grad():
         xTest=torch.from_numpy(xTest)
         post_f = model(xTest)
         post_obs = likelihood(post_f, xTest)
    return post_f,post_obs



#////////////////////////////////////////////////////
def gprTorch_2d_singleTask(xTrain,yTrain,noiseSdev,xTest):
    """ 
        GPR for two uncertain parameter, and 1-D response y, where y=f(x)+e, with Known noise
        - Observations (X_i,Y_i) are assumed to be independent but their noise variance can be either the same (iid= homoscedastic) or different (heteroscedastic).
        - Supports both homo- and hetero-scedastic noise models

        Inputs:          
               xTrain: Training model input, 2D numpy array of size nx2
               yTrain: Training model output, 1D numpy array of size n
               noiseSdev: A 1D numpy vecor of size n, standard deviation of the the Gaussian noise in each of the observations
               xTest: Test model input, 2D numpy array of size nTestx2
        Outputs: 
               post_f: posterior gpr for f(q) at qTest
               post_obs: predictive posterior (likelihood) at qTest
    """
    #---------- SETTING
    nIter=500   #number of iterations in optimization of GPR hyperparameters
    #-------------------------------
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
    #likelihood = GaussianLikelihood(noise=noiseSdev**2.)   #common Gaussian likelihood with no inference for noise levels

    #  (b) prior GPR model
    model = SingletaskGPModel_mIn(xTrain, yTrain, likelihood)

    #(3) Optimize the hyperparameters
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
                {'params': model.parameters()} # Includes GaussianLikelihood parameters
                ], lr=0.0035)   #lr: learning rate
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(nIter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(xTrain)
        # Calc loss and backprop gradients
        loss = -mll(output, yTrain, xTrain)
        #loss = -mll(output, yTrain)
        loss.backward()
        print('...... GPR-hyperparameters Optimization, iter %d/%d - loss: %.3f lengthscale=%.3f %.3f' % (
             i + 1, nIter, loss.item(),
#             model.covar_module.base_kernel.lengthscale.item()   #if all lengthscales are the same (see the definition of self.covar_module, above)
             model.covar_module.base_kernel.lengthscale.squeeze()[0].item(), #different length scales in each dimension
             model.covar_module.base_kernel.lengthscale.squeeze()[1].item(),
             ))
        optimizer.step()
        print('lr=',optimizer.param_groups[0]['lr'])
    #(4) Posteriors of GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()
    #(3) Prediction at test inputs
    #xTest = gpytorch.utils.grid.create_data_from_grid(testGrid)    #torch
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
    #-------------------------------
    #generate training data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
    gprTorch_1d(xTrain,yTrain,noiseSdev)

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
    n=128   #number of training data
    nTest=100   #number of test data
    xBound=[0.,1]   #range of input
    #type of the noise in the data
    noiseType='hetero'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    #-------------------------------
    #(1) Generate training and test data
    xTrain,yTrain,noiseSdev=trainData(xBound,n,noiseType)
    xTest = np.linspace(xBound[0]-0.2, xBound[1]+.2, nTest) #if numpy is used for training
    #xTest = torch.linspace(xBound[0], xBound[1], nTest)   #if torch is used for training
     
    #(2) Construct the GPR using the training data
    post_f,post_obs=gprTorch_1d(xTrain,[yTrain],noiseSdev,xTest)

    #(3) Plots
    with torch.no_grad():
         lower_f, upper_f = post_f.confidence_region()
         lower_obs, upper_obs = post_obs.confidence_region()
         plt.figure(figsize=(10,6))
         plt.plot(xTest,fEx(xTest),'--b')
         plt.plot(xTrain, yTrain, 'ok',markersize=4)
         plt.plot(xTest, post_f.mean[:].numpy(), '-r',lw=2)
         plt.fill_between(xTest, lower_f.numpy(), upper_f.numpy(), alpha=0.3)
         plt.fill_between(xTest, lower_obs.numpy(), upper_obs.numpy(), alpha=0.15, color='r')
         plt.legend(['Exact Reponse','Observed Data', 'Mean Prediction', 'Confidence (f)', 'Confidence (obs)'],loc='best',fontsize=15)
         #NOTE: confidence = 2* sdev, see 
#         https://github.com/cornellius-gp/gpytorch/blob/4a1ba02d2367e4e9dd03eb1ccbfa4707da02dd08/gpytorch/distributions/multivariate_normal.py
         plt.title('Single-Task GP + Heteroscedastic Noise')
         plt.xticks(fontsize=18)
         plt.yticks(fontsize=18)
         plt.xlabel(r'$\mathbf{q}$',fontsize=17)
         plt.ylabel(r'$y$',fontsize=17)
         plt.show()

#//////////////////////////////////
def gprTorch_2d_singleTask_test():
    """
        Test for GPR for 2d input
    """
    def test_2dGrid(bounds1,bounds2,nTest1,nTest2):
        """
           Construct a 2D mesh for test inputs defined on ranges bounds1,bounds2 as equi-spaced points.
           The mesh is to be used for plot the contours in 2D plane. 
        """
        nTest=nTest1*nTest2
        x1Test=np.linspace(bounds1[0],bounds1[1],nTest1)
        x2Test=np.linspace(bounds2[0],bounds2[1],nTest2)
        x1TestGrid=np.zeros((nTest1,nTest2))
        x2TestGrid=np.zeros((nTest1,nTest2))
        xTestArr=np.zeros((nTest,2));
        for i in range(nTest1):
            for j in range(nTest2):
               k=i+j*nTest1
               xTestArr[k,0]=x1Test[i]
               xTestArr[k,1]=x2Test[j]
               x1TestGrid[i,j]=x1Test[i]
               x2TestGrid[i,j]=x2Test[j]
        xTestArr=np.asarray(xTestArr)   #n* x p=2
        return x1TestGrid,x2TestGrid,xTestArr
    def noiseGen(n,noiseType):
       """
          Generate a 1D numpy array of standard deviations of independent Gaussian noises
       """
       if noiseType=='homo':
          sd=5   #standard deviation (NOTE: cannot be zero)
          sdV=[sd]*n
          sdV=np.asarray(sdV)
       elif noiseType=='hetero':
          sdMin=5
          sdMax=20.0
          sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
       return sdV  #vector of standard deviations

    #----- SETTINGS
    qBound=[[-2,2],[-2,2]]
    sampleType='grid'  #'random' or 'grid': type of samples
    noiseType='homo'   #'homo'=homoscedastic, 'hetero'=heterscedastic
    #------------------------------------------------
    #(1) Generate training data
    d=len(qBound)    #dimension of the input
    #  (a) xTrain 
    if sampleType=='grid':
       n=[9,5]             #number of training observations in each input dimension
       nSamp=n[0]*n[1]
       gridList=[];
       for i in range(d):
           #grid_=torch.linspace(qBound[i][0],qBound[i][1],n[i])   #torch
           grid_=np.linspace(qBound[i][0],qBound[i][1],n[i])
           gridList.append(grid_)
       xTrain=reshaper.vecs2grid(gridList[0],gridList[1])
#       xTrain = gpytorch.utils.grid.create_data_from_grid(gridList)  #torch
    elif sampleType=='random': 
       nSamp=40     #number of random samples   
       xi_=sampling.LHS_sampling(nSamp,d)
       xTrain=np.zeros((nSamp,d))
       for i in range(d):
           xTrain[:,i]=(qBound[i][1]-qBound[i][0])*xi_[:,i]+qBound[i][0]
    #   (b) set the sdev of the observation noise   
#    noiseSdev=torch.ones(nTot).mul(0.1)    #torch
    noiseSdev=noiseGen(nSamp,noiseType)
#    yTrain = torch.sin(mt.pi*xTrain[:,0])*torch.cos(.25*mt.pi*xTrain[:,1])+torch.randn_like(xTrain[:,0]).mul(0.1)   #torch
    #   (c) Training data
    yTrain=analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],'Rosenbrock','pair')
    yTrain_mean=0.0#np.mean(yTrain)    #to centeralize the yTrain data
    yTrain=yTrain-yTrain_mean
    yTrain=yTrain+noiseSdev*np.random.randn(nSamp)

    #(2) Create test data
    testGrid=[];
    nTest=[100,100]     #number of test points
    for i in range(d):
        #grid_=torch.linspace(qBound[i][0],qBound[i][1],20)    #torch
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        testGrid.append(grid_)
    xTest=reshaper.vecs2grid(testGrid[0],testGrid[1])

    #(3) construct the GPR based on the training data and make predictions at test inputs
    post_f,post_obs=gprTorch_2d_singleTask(xTrain,yTrain,noiseSdev,xTest)

    #(4) Plot 2d contours
    #   (a) Make a test grid
    x1TestGrid_,x2TestGrid_,xTestArr=test_2dGrid(qBound[0],qBound[1],nTest[i],nTest[i])
    #   (b) Predicted mean and variance at the test grid
    post_f_mean=post_f.mean.reshape(x1TestGrid_.shape).T
    lower_f, upper_f = post_f.confidence_region()
    lower_f=lower_f.reshape(x1TestGrid_.shape).T
    post_f_sdev = (post_f_mean-lower_f)/2.0   #posterior sdev of f(q)
    with torch.no_grad():
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(131)        
        fEx_test=analyticTestFuncs.fEx2D(xTestArr[:,0],xTestArr[:,1],'Rosenbrock','pair')
        CS0=ax.contour(x1TestGrid_,x2TestGrid_,fEx_test.reshape(x1TestGrid_.shape,order='F'),levels=40)
        ax.clabel(CS0, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Exact $f(q)$')
        ax = fig.add_subplot(132)
        CS1=ax.contour(x1TestGrid_,x2TestGrid_,post_f_mean.detach().numpy()+yTrain_mean,levels=40)
        ax.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Mean Posterior of $f(q)$')
        ax = fig.add_subplot(133)
        CS2=ax.contour(x1TestGrid_,x2TestGrid_,post_f_sdev.detach().numpy(),levels=40)
        ax.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        ax.plot(xTrain[:,0],xTrain[:,1],'or')
        ax.set_title(r'Sdev of Posterior of $f(q)$')
        plt.show()
