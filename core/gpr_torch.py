################################################################
# Gaussian Process Regression (GPR)
# Using gpytorch library
#################################################################
#----------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------------
"""
Notes:
  1. GPyTorch: Size of the training data cannot exceed 128.
     Otherwise, make batches of max size 128.

  2. GPyTorch: If the standard deviation of the observation noise is 
  exactly zero for all observations, then there may be issues with
  Cholesky decomposition. Instead of zero, use a very small value for
  the noise standard deviations.

  3. In a p-D parameter space, it is required to define a length-scale
  per dimension. Based on the experinece, if the range of the parameters
  are too different from each other or are too large, the optimization
  of the length-scales can be problematic. To rectify this, the original
  parameter space can be mapped, for instance, into the hypercube
  [-1,1]^p.

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
sys.path.append(os.getenv("UQit"))
import analyticTestFuncs
import reshaper
import sampling
#
class SingletaskGPModel(gpytorch.models.ExactGP):
    """
    GPR for single-task output, 1D input: y=f(x) in R, x in R
    Based on `GPyTorch`
    """
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
#
class SingletaskGPModel_mIn(gpytorch.models.ExactGP):
    """
    GPR for single-task output, p-D input: y=f(x)  in R, x in R^p, p>1
    Based on `GPyTorch`
    """
    def __init__(self, train_x, train_y, likelihood):
        super(SingletaskGPModel_mIn, self).__init__(train_x, train_y, likelihood)
        num_dims = train_x.size(-1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            #gpytorch.kernels.RBFKernel(ard_num_dims=num_dims)   
            ##different length scales in different dimentions, RBF
            gpytorch.kernels.MaternKernel(nu=2.5,ard_num_dims=num_dims)   
            ##different length scales in different dimentions, Matern nu
            #gpytorch.kernels.RBFKernel()   #equal length scales in all input dimensions
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#
class gpr:
    """
    Constructs and evaluates a GPR over the space of uncertain parameters/inputs. The GPR model is
    
    .. math::
       y=f(x)+\epsilon 
       
    where :math:`\epsilon~N(M,V)`

     * The parameter (input) space is p-dimensional, where p=1,2,...,p.
     * The response y is 1-D (single-task model).
     * The observations are assumed to be un-correlated, but the noise uncertainty 
       can be observation-dependent (heteroscedastic). Therefore, only the diagonal
       elements of V are (currently) assumed to be non-zero. If the noise uncertainty 
       is fixed for all observations, then e_i~iid (homoscedastic).

    Args:          
      `xTrain`: 2D numpy array of size nxp
         Training samples (size n) from the p-D input (parameter) space
      `yTrain`: 2D numpy array of size nxm (m: dimensionality of y)
         Training model output (response)
      `noiseV`: A 1D numpy vecor of size n 
         Standard deviation of the the Gaussian noise (diagonal elements of V)
      `xTest`: 2D numpy array of size nTestxp
         Test samples from the input (parameter) space
      `gprOpts`: dict
         Options for constructing GPR with the following keys:
           * 'nIter': int
             Number of iterations in the optimization of hyperparameters
           * 'lr': float 
             Learning rate for the optimizaer of the hyperparameters
           * 'convPlot': bool 
             If true, optimized values of the hyper-parameters is plotted vs. iteration.

    Attributes: 
      `post_f`: Posterior gpr for f(x) at `xTest`
      `post_obs`: Predictive posterior (likelihood) at `xTest`
    """
    def __init__(self,xTrain,yTrain,noiseV,xTest,gprOpts):
        self.xTrain=xTrain
        self.yTrain=yTrain
        self.noiseV=noiseV
        self.xTest=xTest
        self.gprOpts=gprOpts
        self._info()
        self.train_pred()
    
    def _info(self):
       self.p=self.xTrain.shape[-1]
       self.nResp=self.yTrain.shape[-1]

    def train_pred(self):    
        """
        Constructor of the GPR
        """
        if self.p==1:
           self.gprTorch_1d() 
        elif self.p>1:
           self.gprTorch_pd() 

    def gprTorch_1d(self):
        """
        GPR for 1D uncertain parameter
        Observations :math:`{(x_i,y_i)}_{i=1}^n` are assumed to be independent but their noise 
        variance can be either the same (iid=homoscedastic) or non-identical (heteroscedastic).
        """ 
        if self.nResp==1:   #single-task (=single-response)
           self.gprTorch_1d_singleTask()
        else:          #multi-task (=multi-response)
           raise ValueError('Multitask version of GPR is not available yet!') 

    def gprTorch_1d_singleTask(self):
        """ 
        GPR for 1D uncertain parameter and single-variate response y.
        """
        xTrain=self.xTrain
        yTrain=self.yTrain[:,0]
        noiseSdev=self.noiseV
        xTest=self.xTest
        gprOpts=self.gprOpts
        #(0) Assignments
        nIter=gprOpts['nIter']   #number of iterations in optimization of hyperparameters
        lr_  =gprOpts['lr']      #learning rate for the optimizaer of the hyperparameters
        torch.set_printoptions(precision=8)  #to avoid losing accuracy in print after converting to torch
        #(1) convert numpy arrays to torch tensors
        xTrain=torch.from_numpy(xTrain)
        yTrain=torch.from_numpy(yTrain)
        yLogVar=torch.from_numpy(np.log(noiseSdev**2.))

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
        self.loss=losses     
        self.lengthSc=lengthSc
        # Plot convergence of hyperparameters optimization
        if gprOpts['convPlot']:
          self.optim_conv_plot()

        #(6) Posteriors of GPR model with optimized hyperparameters
        model.eval()
        likelihood.eval()  
        #(7) Evaluate the posteriors at the test points
        with torch.no_grad():
            xTest=torch.from_numpy(xTest)
            post_f = model(xTest)
            post_obs = likelihood(post_f, xTest)
        self.post_f=post_f    
        self.post_y=post_obs

    def gprTorch_pd(self):
        """ 
        GPR for p-D (p>1) uncertain parameter.
        Observations (X_i,Y_i) are assumed to be independent but their noise variance can 
        be either the same (iid=homoscedastic) or different (heteroscedastic).
        """
        if self.nResp==1:   #single-task (=single-response)
           self.gprTorch_pd_singleTask()
        else:          #multi-task (=multi-response)
           raise ValueError('Multitask version of GPR is not available yet!') 

    def gprTorch_pd_singleTask(self):
        """ 
        GPR for p>1 uncertain parameter and single-variate response y
        """
        xTrain=self.xTrain
        yTrain=self.yTrain[:,0]
        noiseSdev=self.noiseV
        xTest=self.xTest
        gprOpts=self.gprOpts
        p=self.p
        #(0) Assignments
        nIter=gprOpts['nIter']   #number of iterations in optimization of hyperparameters
        lr_  =gprOpts['lr']      #learning rate for the optimizaer of the hyperparameters
        torch.set_printoptions(precision=8)  #to avoid losing accuracy in print after converting to torch
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
        # likelihood = GaussianLikelihood(noise=noiseSdev**2.)   
        ##common Gaussian likelihood with no inference for heteroscedastic noise levels

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
                #lengthSC_.append(model.covar_module.base_kernel.lengthscale.item())   
                ##if all lengthscales are the same (see the definition of self.covar_module, above)
            if (i+1) % 100 == 0:
               print('...... GPR-hyperparameters Optimization, iter %d/%d - loss: %.3f' %((i + 1), nIter, loss_),end="  ")
               print('lengthscales='+'%.3f '*p %(tuple(lengthSc_)))
            losses.append(loss_)
            for j in range(p):
                lengthSc[j].append(lengthSc_[j])
        self.loss=losses    
        self.lengthSc=lengthSc
        #print('lr=',optimizer.param_groups[0]['lr'])
        #print('pars',optimizer.param_groups[0]['params'])
        # Plot convergence of hyperparameters optimization
        if gprOpts['convPlot']:
           self.optim_conv_plot()
        #(4) Posteriors of GPR model with optimized hyperparameters
        model.eval()
        likelihood.eval()
        #(3) Prediction at test inputs
        with torch.no_grad():
            xTest=torch.from_numpy(xTest)
            post_f = model(xTest)
            post_obs = likelihood(post_f, xTest)
            self.post_f=post_f
            self.post_y=post_obs

    def optim_conv_plot(self):
        """
        Plot convergence of loss and length-scale 
        """
        plt.figure(figsize=(12,3))
        plt.subplot(121)
        plt.plot(self.loss,'-r')   
        plt.ylabel('Loss',fontsize=16)
        plt.xlabel('Iteration',fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.subplot(122)
        if self.p==1:
           plt.plot(self.lengthSc,'-b')
        elif self.p>1:
           for j in range(self.p):
               plt.plot(self.lengthSc[j],'-',label='Lengthscale, param%d'%(j+1))
        plt.ylabel('Lengthscale',fontsize=16)
        plt.xlabel('Iteration',fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        if self.p>1:
           plt.legend(loc='best',fontsize=14)
        plt.suptitle("Convergence of GPR's hyperparam optimization")
        plt.show()

class gprPost:
    """
    Post-processing a constructed GPR

    Args:
      `gpPost`: GpyTorch object
         GPR posterior density created by GPyTorch
      `nTest`: A list of size p
         Containing number of test points in each parameter dimension: [nTest1,nTest2,...nTestp]

    Methods:
      `torchPost()`: Computing mean, standard-deviation, and CI of the GPR-posterior

    Attributes:
      `mean`: p-D numpy array of size (nTest1,nTest2,...,nTestp)
         Mean of the GP-posterior
      `sdev`: pD numpy array of size (nTest1,nTest2,...,nTestp)
         Standard-deviation of the GP-posterior
      `ciL`: pD numpy array of size (nTest1,nTest2,...,nTestp)
         Lower 95% CI of the GP-posterior
      `ciU`: pD numpy array of size (nTest1,nTest2,...,nTestp)
         Upper 95% CI of the GP-posterior
    """
    def __init__(self,gpPost,nTest):
        self.gpPost=gpPost
        self.nTest=nTest

    def torchPost(self):
        """
        Computes mean, standard-deviation, and CI of the GPR-posterior created by GPyTorch
        """
        with torch.no_grad():
            post_=self.gpPost
            nTest=self.nTest
            post_mean_=post_.mean.detach().numpy()
            post_mean =post_mean_.reshape(nTest,order='F')   #posterior mean
            lower_, upper_ = post_.confidence_region()     #\pm 2*sdev of posterior mean
            lower_=lower_.detach().numpy().reshape(nTest,order='F')
            upper_=upper_.detach().numpy().reshape(nTest,order='F')
            post_sdev = (post_mean-lower_)/2.0   #sdev of the posterior mean of f(q)
        self.mean=post_mean    
        self.sdev=post_sdev
        self.ciL=lower_
        self.ciU=upper_

class gprPlot:
    """
    Plotters for GPR

    Args:
      `pltOpts`: dict (optional) 
       Options for plane plots (p=1 or 2) with the following keys:
         * 'title': string, plot title
         * 'legFS': float, legend fontsize
         * 'labFS': [float,float], x,y-axes label fontsize
         * 'ticksFS': [float,float], x,y-ticks fontsize
         * 'save': bool   
            If 'save':True, then
              - 'figName': string, figure name 
              - 'figDir': string, directory to save the fig
              - 'figSize': [float,float], figure size

    Methods:    
       `torch1d()`: Single-task GPR on a 1d array
       `torch2d_2dcont()`:
       `torch2d_3dSurf()`:
    """
    def __init__(self,pltOpts={}):
        self.pltOpts=pltOpts
        self._set_vals()

    def _set_vals(self):
        keys_=self.pltOpts.keys() #provided keys
        if 'title' in keys_:
           self.title=self.pltOpts['title']
        if 'xlab' in keys_:
           self.xlab=self.pltOpts['xlab']
        if 'ylab' in keys_:
           self.ylab=self.pltOpts['ylab']
        #default values for fontsizes
        self.titleFS=15        #title fs
        self.legFS=15          #legend fs
        self.labFS=[17,17]     #axes-label fs
        self.ticksFS=[18,18]   #axes-label fs
        if 'titleFS' in keys_:
           self.titleFS=self.pltOpts['titleFS']
        if 'legFS' in keys_:
           self.legFS=self.pltOpts['legFS']
        if 'labFS' in keys_:
           self.labFS=self.pltOpts['labFS']
           if len(self.labFS)!=2:
              raise ValueError("Value of 'labFS' should have length 2 (x,y-axes label fontsize).")
        if 'ticksFS' in keys_:
           self.ticksFS=self.pltOpts['ticksFS']
           if len(self.ticksFS)!=2:
              raise ValueError("Value of 'ticksFS' should have length 2 (x,y-axes label fontsize).")
        #options for saving the plot        
        self.figSave=False
        if 'save' in keys_ and self.pltOpts['save']:
           list_=['figName','figDir','figSize'] 
           for L_ in list_:
               if L_ not in keys_:
                  raise KeyError("%s is required in pltOpts since 'save' is True." %L_)
               else:
                  globals()[L_]=self.pltOpts[L_]
           self.figName=figName       
           self.figDir=figDir
           self.figSize=figSize
           self.figSave=True
    
    def _figSaver(self):
        """
        Saves figures
        """
        fig = plt.gcf()
        if not os.path.exists(self.figDir):
           os.makedirs(self.figDir)
        figSave_=figDir+figName
        figOut=figDir+figName
        DPI = fig.get_dpi()
        fig.set_size_inches(self.figSize[0]/float(DPI),self.figSize[1]/float(DPI))
        plt.savefig(figOut+'.pdf',bbox_inches='tight')
                      
    def torch1d(self,post_f,post_obs,xTrain,yTrain,xTest,fExTest):
        """
        Plots the GPR constructed by GPyToch for a 1D input in the input space.
      
        Args:
          `post_f`: GpyTorch object
             Posterior density of model function f(q)
          `post_obs`: GpyTorch object
             Posterior density of the response y
          `xTrain`: 1D numpy array of size nTrain
             GPR training samples taken from the input space
          `fExTest`: 1D numpy array of size nTrain
             Response values at `xTrain`
          `xTest`: 1D numpy array of size nTest
             Test samples taken from the input space
          `fExTest`: 1D numpy array of size nTest
             Exact response values at `xTest`
        """
        with torch.no_grad():
             lower_f, upper_f = post_f.confidence_region()
             lower_obs, upper_obs = post_obs.confidence_region()
             plt.figure(figsize=(10,6))
             plt.plot(xTest,fExTest,'--b',label='Exact Output')
             plt.plot(xTrain, yTrain, 'ok',markersize=4,label='Training observations')
             plt.plot(xTest, post_f.mean[:].numpy(), '-r',lw=2,label='Mean Model')
             plt.plot(xTest, post_obs.mean[:].numpy(), ':m',lw=2,label='Mean Posterior Prediction')
             plt.plot(xTest, post_obs.sample().numpy(), '-k',lw=1,label='Sample Posterior Prediction')
             plt.fill_between(xTest, lower_f.numpy(), upper_f.numpy(), alpha=0.3,label='Confidence f(q)')
             plt.fill_between(xTest, lower_obs.numpy(), upper_obs.numpy(), alpha=0.15, color='r',label='Confidence Yobs')
             plt.legend(loc='best',fontsize=self.legFS)
             #NOTE: confidence = 2* sdev, see
             #https://github.com/cornellius-gp/gpytorch/blob/4a1ba02d2367e4e9dd03eb1ccbfa4707da02dd08/gpytorch/distributions/multivariate_normal.py             
             if hasattr(self,'title'):
                plt.title(self.title,fontsize=self.titleFS) 
             else:   
                plt.title('Single-task GP, 1D parameter',fontsize=self.titleFS)
             plt.xticks(fontsize=self.ticksFS[0])
             plt.yticks(fontsize=self.ticksFS[1])
             plt.xlabel(r'$\mathbf{q}$',fontsize=self.labFS[0])
             plt.ylabel(r'$y$',fontsize=self.labFS[1])             
             if self.figSave:
                self._figSaver()
             plt.show()

    def torch2d_2dcont(self,xTrain,qTest,fTestGrid):
        """
        Planar contour plots of a GPR constructed in a 1D input space.

        Args:
          `xTrain`: 2D numpy array of shape (nTrain,2)
             GPR training samples taken from the input space
          `yTrain`: 1D numpy array of size nTrain
             Response values at `xTrain`
          `qTest`: List of length 2
             =[qTest_1,qTest2], qTest_i: 1D array of size nTest_i of the test 
              samples taken from the space of i-th input
          `fTestGrid`: 2D numpy array of shape (nTest_1,nTest_2)    
              Response values at a tensor-product grid constructed from `qTest`
        """
        fig = plt.figure(figsize=(5,5))
        plt.plot(xTrain[:,0],xTrain[:,1],'or')
        CS1=plt.contour(qTest[0],qTest[1],fTestGrid.T,levels=40)
        plt.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
        plt.legend(loc='best',fontsize=self.legFS)
        plt.xticks(fontsize=self.ticksFS[0])
        plt.yticks(fontsize=self.ticksFS[1])
        if hasattr(self,'xlab'):
           plt.xlabel(self.xlab,fontsize=self.labFS[0]) 
        if hasattr(self,'ylab'):
           plt.ylabel(self.ylab,fontsize=self.labFS[1]) 
        if hasattr(self,'title'):
           plt.title(self.title,fontsize=self.titleFS) 
        if self.figSave:
           self._figSaver()
        plt.show()
             
    def torch2d_3dSurf(self,xTrain,yTrain,qTest,post_obs,post_f):
        """
        3D plot of the GPR surface (mean+CI) constructed for a 2D input (parameter).

        Args:
          `xTrain`: 2D numpy array of shape (nTrain,2)
             GPR training samples taken from the input space
          `yTrain`: 1D numpy array of size nTrain
             Response values at `xTrain`
          `qTest`: List of length 2
             =[qTest_1,qTest2], qTest_i: 1D array of size nTest_i of the test 
              samples taken from the space of i-th input
          `post_f`: GpyTorch object
             Posterior density of model function f(q)
          `post_obs`: GpyTorch object
             Posterior density of the response y
        """
        nTest=[len(qTest[i]) for i in range(len(qTest))]
        #Predicted mean and variance at the test grid
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

        xTestGrid1,xTestGrid2=np.meshgrid(qTest[0],qTest[1], sparse=False, indexing='ij')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        mean_surf = ax.plot_surface(xTestGrid1, xTestGrid2, post_obs_mean,cmap='jet', antialiased=True,rstride=1,cstride=1,linewidth=0,alpha=0.4)
        upper_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, upper_obs, linewidth=1,alpha=0.25,color='r')
        lower_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, lower_obs, linewidth=1,alpha=0.25,color='b')
        #upper_surf_f = ax.plot_wireframe(xTestGrid1, xTestGrid2, upper_f, linewidth=1,alpha=0.5,color='r')
        #lower_surf_f = ax.plot_wireframe(xTestGrid1, xTestGrid2, lower_f, linewidth=1,alpha=0.5,color='b')
        plt.plot(xTrain[:,0],xTrain[:,1],yTrain,'ok')
        if self.figSave:
           self._figSaver()
        plt.show()
#
#
# TESTS
#
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
      
    #----- SETTINGS ----------------
    n=120   #number of training data
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
    gpr_=gpr(xTrain=xTrain[:,None],yTrain=yTrain[:,None],noiseV=noiseSdev,xTest=xTest[:,None],gprOpts=gprOpts)
    post_f=gpr_.post_f
    post_obs=gpr_.post_y

    #(3) Plots
    fExTest=fEx(xTest)
    #plot options (none is mandatory)
    pltOpts={'title':'Single-task GP, 1d param, %s-scedastic noise'%noiseType}
    gprPlot(pltOpts).torch1d(post_f,post_obs,xTrain,yTrain,xTest,fExTest)
#
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
          xTrain=reshaper.vecs2grid(gridList)
#       xTrain = gpytorch.utils.grid.create_data_from_grid(gridList)  #torch
        elif sampleType=='random': 
             nSamp=n     #number of random samples   
             xTrain=sampling.LHS_sampling(n,qBound)
        #  (b) set the sdev of the observation noise   
        #noiseSdev=torch.ones(nTot).mul(0.1)    #torch
        noiseSdev=noiseGen(nSamp,noiseType,xTrain,fExName)
        #yTrain = torch.sin(mt.pi*xTrain[:,0])*torch.cos(.25*mt.pi*xTrain[:,1])+torch.randn_like(xTrain[:,0]).mul(0.1)   #torch
        #   (c) Training data
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
          #sdMin=0.01
          #sdMax=0.5
          #sdV=sdMin+(sdMax-sdMin)*np.linspace(0.0,1.0,n)
          #sdV=0.15*np.ones(n)
          sdV=0.1*(analyticTestFuncs.fEx2D(xTrain[:,0],xTrain[:,1],fExName,'comp').val+0.001)
       return sdV  #vector of standard deviations

    #----- SETTINGS
    qBound=[[-2,2],[-2,2]]   #Admissible range of parameters
    #options for training data
    fExName='type2'          #Name of Exact function to generate synthetic data
                             #This is typ in fEx2D() in ../../analyticFuncs/analyticFuncs.py
    sampleType='random'        #'random' or 'grid': type of samples
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
    xTestList=[];
    for i in range(p):
        #grid_=torch.linspace(qBound[i][0],qBound[i][1],20)    #torch
        grid_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        xTestList.append(grid_)
    xTest=reshaper.vecs2grid(xTestList)

    #(3) construct the GPR based on the training data and make predictions at test inputs
    gpr_=gpr(xTrain,yTrain[:,None],noiseSdev,xTest,gprOpts)
    post_f=gpr_.post_f
    post_obs=gpr_.post_y

    #(4) Plot 2d contours
    #   (a) Predicted mean and variance at the test grid    
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
    #   (b) Plots
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
        gprPlot().torch2d_3dSurf(xTrain,yTrain,xTestList,post_obs,post_f)
#
