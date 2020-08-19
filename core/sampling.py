#############################################
#       Sampling methods
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
#
import sys
import numpy as np
import math as mt
import nodes
import pce
#
#
#//////////////////////////
def LHS_sampling(n,range_):
    """
        LHS (Latin Hypercube) sampler from a p-D random variable distributed uniformly
        credit: https://zmurchok.github.io/2019/03/15/Latin-Hypercube-Sampling.html
        Inputs:
              n: number of samples
              range_: Admissible orange of the samples,list of size p, =[[x1L,x1U],...[xpL,xpU]]
        Outputs:
              x: n-by-p numpy array, x\in range_
    """
    p=len(range_)
    x = np.random.uniform(size=[n,p])
    for i in range(0,p):
        x_ = (np.argsort(x[:,i])+0.5)/float(n)
        x[:,i]=x_*(range_[i][1]-range_[i][0])+range_[i][0]
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
                 'random', 'uniform', 'GL', 'Clenshaw', 'Clenshaw-Curtis'
    Outputs:
       qNodes: samples of size nSamp over range_ 
    """
    p=len(range_)
    xi_len=1.
    xi0=0.
    if sampType=='random':
       xi=np.random.uniform(0,1,size=[nSamp-2])
       xi=np.concatenate([[0],xi,[1]])
    elif sampType =='uniform':
       xi=np.linspace(0,1,nSamp)
    elif sampType =='GL':
       xi,wXI=pce.gqPtsWts(nSamp,'Unif')  #on [-1,1]
       xi_len=2.
       xi0=-1.
    elif sampType=='Clenshaw':
       xi=nodes.Clenshaw_pts(nSamp)
       xi_len=2.
       xi0=-1.
    elif sampType=='Clenshaw-Curtis':
       l_=1+int(mt.log(nSamp-1)/mt.log(2))
       xi=nodes.ClenshawCurtis_pts(l_)
    else:
       print('ERROR in sampler_1d(): Invalid sampType was chosen!')
    #map from reference range to actual range
    qNodes=(range_[1]-range_[0])*(xi-xi0)/xi_len+range_[0]
    return qNodes



