#######################################################################
#    Different nodes to be used in quadrature rules and interpolation
#######################################################################
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------------------
#
import numpy as np
import math as mt
#
#
def Clenshaw_pts(n):
    """ 
       Returns n Clenshaw points over [-1,1]
    """
    x=np.zeros(n)
    for i in range(n):
        x[i]=np.cos(i*mt.pi/(n-1))
    x_=x[::-1]    
    return x_
#
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
#
def gllPts(n,eps=10**-8,maxIter=1000):
    """
    Iterative solution to get Gauss-Lobatto-Legendre (GLL) nodes of order n
    Newton-Raphson iteration is used.

    Parameters
    ----------
    n: number of GLL nodes
    eps (optional) iteration max error
    maxIter (optional): max iteration

    Outputs
    -------
    xi: 1d numpy array of size n, GLL nodes
    w: 1d numpy array of size n, GLL weights

    Reference
    ---------
    Canuto C., Hussaini M. Y., Quarteroni A., Tang T. A., "Spectral Methods
       in Fluid Dynamics," Section 2.3. Springer-Verlag 1987.
       https://link.springer.com/book/10.1007/978-3-642-84108-8
    """
    V=np.zeros((n,n))  #Legendre Vandermonde Matrix
    #Initial guess for the nodes: Clenshaw points
    xi=Clenshaw_pts(n)
    iter_=0
    err=1000
    xi_old=xi
    while iter_<maxIter and err>eps:
        iter_+=1
        #Update the Legendre-Vandermonde matrix
        V[:,0]=1.
        V[:,1]=xi
        for j in range(2,n):
            V[:,j]=((2.*j-1)*xi*V[:,j-1] - (j-1)*V[:,j-2])/float(j)
        #Newton-Raphson solver
        xi=xi_old-(xi*V[:,n-1]-V[:,n-2])/(n*V[:,n-1])
        err=max(abs(xi-xi_old).flatten())
        xi_old=xi
    #Weights
    w=2./(n*(n-1)*V[:,n-1]**2.)
    return xi,w

