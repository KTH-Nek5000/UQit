#############################################
#       Sampling methods
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
#
import sys
import numpy as np
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


