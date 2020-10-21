################################################################
# Gaussian Process Regression (GPR)
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
  Cholesky decomposition. Therefore, instead of zero, use a very small 
  value for the noise standard deviations.

  3. In a p-D parameter space, it is required to define a length-scale
  per dimension. Based on the experience, if the range of the parameters
  are too different from each other or are too large, the optimization
  of the length-scales can be problematic. To rectify this, the original
  parameter space can be mapped, for instance, into the hypercube
  [-1,1]^p. Then, the GPR can be constructed on this mapped space. 

"""    
#----------------------------------------------------------------
#
import os
import sys
import numpy as np
import math as mt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #for 3d plot
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
#
class SingletaskGPModel(gpytorch.models.ExactGP):
    """
    GPR for single-task output using GPyTorch, 
    1D input: y=f(x) in R, x in R
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
    GPR for single-task output using GPyTorch, 
    p-D input: y=f(x)  in R, x in R^p, p>1
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
       
    where :math:`\epsilon\sim N(M,V)`

     * The parameter (input) space is p-dimensional, where p=1,2,...
     * The response y is 1-D (single-task model).
     * The observations are assumed to be un-correlated, but the noise uncertainty 
       can be observation-dependent (heteroscedastic). Therefore, only the diagonal
       elements of V are (currently) assumed to be non-zero. If the noise uncertainty 
       is fixed for all observations, then epsilon_i~iid (homoscedastic).

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
               Learning rate in the optimization of hyperparameters
           * 'convPlot': bool 
               If true, optimized values of the hyper-parameters is plotted vs. iteration.

    Attributes: 
      `post_f`: Posterior of f(x) at `xTest`.
      
      `post_obs`: Predictive posterior (likelihood) at `xTest`.
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
        Constructor of the GPR (training and predicting at test samples)
        """
        if self.p==1:
           self.gprTorch_1d() 
        elif self.p>1:
           self.gprTorch_pd() 

    def gprTorch_1d(self):
        """
        GPR for 1D uncertain parameter.
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
        Plot convergence of loss and length-scale during the optimization
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
       Options for planar plots (p=1 or 2) with the following keys:
         * 'title': string, plot title
         * 'legFS': float, legend fontsize
         * 'labFS': [float,float], x,y-axes label fontsize
         * 'ticksFS': [float,float], x,y-ticks fontsize
         * 'save': bool   
            If 'save'==True, then
              - 'figName': string, figure name 
              - 'figDir': string, directory to save the figure
              - 'figSize': [float,float], figure size

    Methods:    
       `torch1d()`: 
          Plots the GPR constructed for a 1D input. 
       `torch2d_2dcont()`: 
          Planar contour plot of a GPR constructed over a 2D input space.
       `torch2d_3dSurf()`: 
          3D plot of the GPR surface (mean+CI) constructed for a 2D input.
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
        Plots the GPR constructed by GPyToch for a 1D input.
      
        Args:
          `post_f`: GpyTorch object
             Posterior density of the model function f(q)
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
             plt.plot(xTrain, yTrain, 'ok',markersize=4,label='Training obs. y')
             plt.plot(xTest, post_f.mean[:].numpy(), '-r',lw=2,label='Mean Model')
             plt.plot(xTest, post_obs.mean[:].numpy(), ':m',lw=2,label='Mean Posterior Pred')
             plt.plot(xTest, post_obs.sample().numpy(), '-k',lw=1,label='Sample Posterior Pred')
             plt.fill_between(xTest, lower_f.numpy(), upper_f.numpy(), alpha=0.3,label='CI for f(q)')
             plt.fill_between(xTest, lower_obs.numpy(), upper_obs.numpy(), alpha=0.15, color='r',label='CI for obs. y')
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
        Planar contour plots of a GPR constructed over a 2D input space.

        Args:
          `xTrain`: 2D numpy array of shape (nTrain,2)
             GPR training samples taken from the input space
          `yTrain`: 1D numpy array of size nTrain
             Response values at `xTrain`
          `qTest`: List of length 2
             =[qTest_1,qTest2], where qTest_i: 1D array of size nTest_i of the test 
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
             
    def torch2d_3dSurf(self,xTrain,yTrain,qTest,post_):
        """
        3D plot of the GPR surface (mean+CI) constructed for a 2D input (parameter).

        Args:
          `xTrain`: 2D numpy array of shape (nTrain,2)
             GPR training samples taken from the input space
          `yTrain`: 1D numpy array of size nTrain
             Response values at `xTrain`
          `qTest`: List of length 2
             =[qTest_1,qTest2], where qTest_i: 1D array of size nTest_i of the test 
             samples taken from the space of i-th input
          `post_`: GpyTorch object
             Posterior density of model function f(q) or observations y
        """
        nTest=[len(qTest[i]) for i in range(len(qTest))]
        #Predicted mean and variance at the test grid
        fP_=gprPost(post_,nTest)
        fP_.torchPost()
        post_mean=fP_.mean
        post_sdev=fP_.sdev
        lower_=fP_.ciL
        upper_=fP_.ciU

        xTestGrid1,xTestGrid2=np.meshgrid(qTest[0],qTest[1], sparse=False, indexing='ij')
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        mean_surf = ax.plot_surface(xTestGrid1, xTestGrid2, post_mean,cmap='jet', 
                antialiased=True,rstride=1,cstride=1,linewidth=0,alpha=0.4)
        upper_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, upper_, linewidth=1,alpha=0.25,color='r')
        lower_surf_obs = ax.plot_wireframe(xTestGrid1, xTestGrid2, lower_, linewidth=1,alpha=0.25,color='b')
        plt.plot(xTrain[:,0],xTrain[:,1],yTrain,'ok',ms='5')
        if self.figSave:
           self._figSaver()
        plt.show()
#
