#
# Single and multi-task GPs with heteroscedastic noise level
#
import sys
import numpy as np
import math 
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
def gprTorch_multiTask():
    """ 
        Multi-Task GPR + heteroscedastic noise level
    """
    print("... Testing heteroschedastic noise - Multi-response case")
    #synthetic data
    train_x = torch.linspace(0, 1, 75)

    sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 75)
    sem_y2 = 0.75 - (0.75 - 0.05) * torch.linspace(0, 1, 75)

    train_y = torch.stack([ 
              torch.sin(train_x * (2 * math.pi)) + sem_y1 * torch.randn(train_x.size()),
              torch.cos(train_x * (2 * math.pi)) + sem_y2 * torch.randn(train_x.size()),
], -1)

    train_y_log_var = torch.stack([(s**2).log() for s in (sem_y1, sem_y2)], -1)
    print(train_y_log_var)
    print('--------------------------------------------------')
    print('xTrain:',train_x.shape)
    print('yTrain:',train_y.shape)    
    print('train_y_log_var:',train_y_log_var.shape)
    print('--------------------------------------------------')
#    print(train_x)
#    print(train_y)
#    print(train_y_log_var)

    #construct the GPR
    numTasks=2

    log_noise_model = MultitaskGPModel(
                      train_x,
                      train_y_log_var,
                      MultitaskGaussianLikelihood(num_tasks=numTasks),
                      num_tasks=numTasks,
                    )
    
    likelihood = _MultitaskGaussianLikelihoodBase(
                 num_tasks=numTasks,
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )

    model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=numTasks, rank=numTasks)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                 ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 75
    for i in range(n_iter):
        optimizer.zero_grad()
        print('after2',train_x.shape)
        output = model(train_x)
        loss = -mll(output, train_y, train_x)
        print('after3',output,train_x.shape,train_y.shape)
        loss.backward()
        if (i+1) % 10 == 0:
           print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():  
         test_x = torch.linspace(0, 1, 35)
         post_f = model(test_x)
         post_obs = likelihood(post_f, test_x)

    with torch.no_grad():
         f, axs = plt.subplots(1, 2, figsize=(14, 6))
         lower_f, upper_f = post_f.confidence_region()
         lower_obs, upper_obs = post_obs.confidence_region()
         for i, ax in enumerate(axs):
             ax.plot(train_x.numpy(), train_y[:, i].numpy(), 'k*')
             ax.plot(test_x.numpy(), post_f.mean[:, i].numpy(), 'b')
             ax.fill_between(test_x.numpy(), lower_f[:, i].numpy(), upper_f[:, i].numpy(), alpha=0.5)
             ax.fill_between(test_x.numpy(), lower_obs[:, i].numpy(), upper_obs[:, i].numpy(), alpha=0.25, color='r')
             ax.set_ylim([-3, 3])
             ax.legend(['Observed Data', 'Mean', 'Confidence (f)', 'Confidence (obs)'])
         plt.title('Multi-Task GP + Heteroscedastic Noise')
         plt.show()

#////////////////////////////////
def gprTorch_singleTask():
    """ 
        Single-Task GPR + heteroscedastic noise level
    """
    print("... Testing heteroschedastic noise - Single-response case")
    #synthetic data
    train_x = torch.linspace(0, 1, 75)
    sem_y1 = 0.05 + (0.55 - 0.05) * torch.linspace(0, 1, 75)
    train_y =torch.sin(train_x * (2 * math.pi)) + sem_y1 * torch.randn(train_x.size())    
    train_y_log_var = (sem_y1**2.).log()

    #model for noise
    log_noise_model = SingletaskGPModel(
                      train_x,
                      train_y_log_var,
                      GaussianLikelihood(),
                    )
    
    likelihood = _GaussianLikelihoodBase(
                 noise_covar=HeteroskedasticNoise(log_noise_model),
               )
    #define the model
    model = SingletaskGPModel(train_x, train_y, likelihood)
    #training the model
    model.train()
    likelihood.train()
    #optimize the model hyperparameters
    optimizer = torch.optim.Adam([  #Adam optimizer: https://arxiv.org/abs/1412.698
                {'params': model.parameters()},  
                ], lr=0.1)

    #"Loss" for GPs - mll: marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    nIter=75
    for i in range(nIter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y, train_x)
        loss.backward()
        if (i+1) % 10 == 0:
           print('...... GPR-hyperParam Optimization, iter %d/%d - loss: %.3f' % (i + 1, nIter, loss.item()))
        optimizer.step()

    # GPR model with optimized hyperparameters
    model.eval()
    likelihood.eval()  

    with torch.no_grad():  
         test_x = torch.linspace(0, 1, 35)
         post_f = model(test_x)
         post_obs = likelihood(post_f, test_x)

    with torch.no_grad():
#         f, axs = plt.subplots(1, 2, figsize=(14, 6))
         lower_f, upper_f = post_f.confidence_region()
         lower_obs, upper_obs = post_obs.confidence_region()
#         for i, ax in enumerate(axs):
         plt.plot(train_x.numpy(), train_y[:].numpy(), 'k*')
         plt.plot(test_x.numpy(), post_f.mean[:].numpy(), 'b')
         plt.fill_between(test_x.numpy(), lower_f.numpy(), upper_f.numpy(), alpha=0.5)
         plt.fill_between(test_x.numpy(), lower_obs.numpy(), upper_obs.numpy(), alpha=0.25, color='r')
         plt.legend(['Observed Data', 'Mean', 'Confidence (f)', 'Confidence (obs)'])
         plt.title('Single-Task GP + Heteroscedastic Noise')
         plt.show()


####################
#MAIN
####################
#multi-task GP
gprTorch_multiTask()

#single-tas GP
gprTorch_singleTask()

