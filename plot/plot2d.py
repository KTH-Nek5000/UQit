##################################################
#   Surface of a function over a 2D space
##################################################
import numpy as np

def plot2D_gridVals(x1,x2,f):
    """
       Reads in numpy arrays
            x1: n1x1
            x2: n2x1
             f: (n1*n2)x1 or n1xn2
       and returns a 2D grid 
            xG1: n1xn2
            xG2: n1xn2
             fG: n1xn2
       Note: in converting f->fG it is always assumed that param2 has the outer loop.
    """
    n1=x1.shape[0] 
    n2=x2.shape[0] 
    xG1=np.zeros((n1,n2))
    xG2=np.zeros((n1,n2))
    fG=np.zeros((n1,n2))
    f_convert=True
    if f.ndim==2 and f.shape[0]==n1 and f.shape[1]==n2:
       fG=f  #in case the imported f is already n1xn2
       f_convert=False

    for i2 in range(n2):
        for i1 in range(n1):
            k=i2*n1+i1
            xG1[i1,i2]=x1[i1]          
            xG2[i1,i2]=x2[i2]          
            if (f_convert):
               fG[i1,i2]=f[k]
    return xG1,xG2,fG

#/////////////////////////
def plot2D_grid(x1,x2):
    """
       Reads in numpy arrays
            x1: n1x1
            x2: n2x1
       and returns a 2D grid 
            xG1: n1xn2
            xG2: n1xn2
    """
    n1=x1.shape[0]       
    n2=x2.shape[0]      
    xG1=np.zeros((n1,n2))
    xG2=np.zeros((n1,n2))
    for i2 in range(n2):
        for i1 in range(n1):
            xG1[i1,i2]=x1[i1]            
            xG2[i1,i2]=x2[i2]  
    return xG1,xG2

