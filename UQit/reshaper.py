#######################################################
# Tools for converting and reshaping arrays and lists
#######################################################
# Saleh Rezaeiravesh, salehr@kth.se
#------------------------------------------------------
#
import numpy as np
#
#
def lengthVector(x):
    """ 
    Returns the length of vector x which can be a list or a numpy array
    """
    if (isinstance(x,np.ndarray)):
       nx=x.size
    elif (isinstance(x,(list))):
       nx=len(x)
    else:
       raise ValueError("Unknown object x")
    return int(nx)
#
def vecs2grid(x):
    """
    Makes a p-D (p>1) tensor-product grid from a list of length p containg 1D numpy arrays of points in each dimension.

    Args:
       `x`: A list of length p>1
          x=[x1,x2,...,xp] where xi is a 1D numpy array of size ni
    
    Returns:
       'z': A numpy array of shape (n1*n2*...*np,p) 
    """
    p=len(x)
    if p<=1:
       raise ValueError("Import a list of length p>1.")
    z_=np.meshgrid(*x,copy=True,sparse=False,indexing='ij')    
    n=z_[-1].size
    z=np.zeros((n,p))
    for i in range(p):
        z[:,i]=z_[i].reshape((n,1),order='F')[:,0]
    return z
#
def vecsGlue(*x):
    """
    Makes a set by gluing p>1 1D numpy arrays x0,x1,...,xp of the same size (=n)

    Args:
       `x`: 1D numpy arrays each having the size of n

    Return:   
       `z`: numpy array of shape (n,p) 
          z[:,i]=xi where i=1,2,...,p
    """
    p=len(x)
    if p<=1:
       raise ValueError("More than one numpy array must be imported.")
    n=lengthVector(x[0])
    for i in range(1,p):
        n_=len(x[i])
        if n_!=n:
           raise ValueError("Imported numpy arrays should have the same size.")
    z=np.zeros((n,p))
    for j in range(p):
       for i in range(n):
           z[i,j]=x[j][i]
    return z
