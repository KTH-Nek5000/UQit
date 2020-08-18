##########################################################
#   Tools for converting, reshaping, ... arrays
##########################################################
# Saleh Rezaeiravesh, salehr@kth.se
#---------------------------------------------------------
#
import numpy as np
#
#
#///////////////////
def lengthVector(x):
    """ 
        Returns length of vector x that is a list or 1D numpy array or ...
    """
    if (isinstance(x,np.ndarray)):
       nx=x.size
    elif (isinstance(x,(list))):
       nx=len(x)
    else:
       print("ERROR in lengthVector: Unknown object x")
    return int(nx)
#
#///////////////////
def vecs2grid(x):
    """
       Make a pD tensor-product grid out of p 1D vectors x 
          x: a p-size list of 1D numpy arrays 
          z: a numpy pD array (n1*n2,...*np,p) 
    """
    p=len(x)
    if p<=1:
       print("ERROR in vecs2grid(): more than one vector should be imported.")
    z_=np.meshgrid(*x,copy=True,sparse=False,indexing='ij')    
    n=z_[-1].size
    z=np.zeros((n,p))
    for i in range(p):
        z[:,i]=z_[i].reshape((n,1),order='F')[:,0]
    return z
#
#///////////////////
def vecsGlue(*x):
    """
       Glue p 1D vectors x0, x1, ...,xp of the same length (= n) together as they are p components of the same nxp numpy array.
          xi: a numpy 1D array or a 1D list of length n, i=1,2,...,p
          z: numpy 2D array (n,p) 
    """
    p=len(x)
    if p<=1:
       print("ERROR in vecsGlue(): more than one vector should be imported.")
    n=lengthVector(x[0])
    for i in range(1,p):
        n_=len(x[i])
        if n_!=n:
           print('ERROR in vecsGlue(): input vectors should be of the same size.')
    z=np.zeros((n,p))
    for j in range(p):
       for i in range(n):
           z[i,j]=x[j][i]
    return z
