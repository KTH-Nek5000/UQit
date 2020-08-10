#############################################
#       Sampling methods
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
#
import sys
import numpy as np
import nodes
#
#
#//////////////////////////
def LHS_sampling(n=10,p=2):
    """
        LHS (Latin Hypercube) sampler from a p-D random variable distributed uniformly
        credit: https://zmurchok.github.io/2019/03/15/Latin-Hypercube-Sampling.html
        Inputs:
              n: number of samples
              p: dimension of the RV
        Outputs:
              x: n-by-p numpy array, x\in[0,1]
    """
    x = np.random.uniform(size=[n,p])
    for i in range(0,p):
        x[:,i] = (np.argsort(x[:,i])+0.5)/float(n)
    return x
#
#
#///////////////////////////////
def sampler_1d(range_,nSamp,sampType):
    """
    Generating samples from a 1D parameter space
    Inputs:
       range_: list of length 2, admissible range of parameters
       nSamp: Number of samples
       sampType: The method to drawing the samples (nodes)
                 'random', 'uniform', 'GL', 'Clenshaw'
    """
    p=len(range_)
    if sampType=='random':
       xi=np.random.uniform(0,1,size=[nSamp])
    elif sampType =='uniform':
       xi=np.linspace(range_[0],range_[1],nSamp)
    elif sampType =='GL':
       xi,wXI=pce.GaussLeg_ptswts(nSamp)  #on [-1,1]
    elif sampType=='Clenshaw':
       xi=nodes.Clenshaw_pts(nSamp)
    #map from reference range to actual range
    qNodes=(range_[1]-range_[0])*xi+range_[0]
    return qNodes



