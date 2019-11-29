#######################################################################
#    Different nodes to be used in quadrature rules and interpolation
#######################################################################
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------------------
import numpy as np
import math as mt

def Clenshaw_pts(n):
    """ 
       Returns n Clenshaw points over [-1,1]
    """
    x=np.zeros(n)
    for i in range(n):
        x[i]=np.cos(i*mt.pi/(n-1))
    return x

#///////////////
def ClenshawCurtis_pts(l):
    """
        Generates Clenshaw-Curtis nodes at level l over [0,1]
    """
    if l>1:
       n=2**(l-1)+1
       x=np.zeros(n)
       for i in range(n):
           x[i]=(1-np.cos(i*mt.pi/(n-1)))/2
    else:
       n=1
       x=np.zeros(n)
       x[0]=0.5
    return x

