##########################################################
#   Tools for converting, reshaping, ... arrays
##########################################################
# Saleh Rezaeiravesh, salehr@kth.se
#---------------------------------------------------------
import numpy as np


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
    return nx


#///////////////////
def vecs2grid(x,y):
    """
       Make a 2D tensor product grid z out of two 1D vectors x and y
          x: a numpy 1D array or a 1D list of length nx
          y: a numpy 1D array or a 1D list of length ny
          z: numpy 2D array (nx*ny,2) 
    """
    nx=lengthVector(x)
    ny=lengthVector(y)
    z=np.zeros((nx*ny,2))
    for iy in range(ny):
        for ix in range(nx):
            iz=iy*nx+ix
            z[iz,0]=x[ix]
            z[iz,1]=y[iy]
    return z

#///////////////////
def vecsGlue(x,y):
    """
       Glue two 1D vectors x and y of the same length together as they are two components of the same entity.
          x: a numpy 1D array or a 1D list of length n
          y: a numpy 1D array or a 1D list of length n
          z: numpy 2D array (n,2) 
    """
    n=lengthVector(x)
    if (n!=lengthVector(y)):
       print('ERROR in vecsGlue: x and y should have the same length')
    z=np.zeros((n,2))
    for i in range(n):
        z[i,0]=x[i]
        z[i,1]=y[i]
    return z

#///////////////////
def vecs2grid3d(x,y,z):
    """
       Make a 3D grid w out of three 1D vectors x, y, z
          x: numpy 1D array of length nx
          y: numpy 1D array of length ny
          z: numpy 1D array of length nz
          w: numpy 2D array (nx*ny*nz,3) 
    """
    nx=x.size
    ny=y.size
    nz=z.size
    w=np.zeros((nx*ny*nz,3))
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                iw=(iz*ny*nx)+(iy*nx)+ix
                w[iw,0]=x[ix]
                w[iw,1]=y[iy]
                w[iw,2]=z[iz]
    return w
