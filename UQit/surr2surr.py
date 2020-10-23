###############################################################
#     Interpolation from a surrogate to another surrogate
###############################################################
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
#
import os
import sys
from UQit.pce import pce, pceEval
from UQit.lagInt import lagInt
import UQit.reshaper as reshaper
#
#
def lagIntAtGQs(fValM1,qM1,spaceM1,nM2,spaceM2,distType):
    """    
    Given response values `fValM1` at `nM1` arbitrary samples over the p-D `spaceM1`, the values at 
    `nM2` Gauss quadrature points over `spaceM2` are computed using Lagrange interpolation. 

       * Both `spaceM1` and `spaceM2` have the same dimension `p`.
       * At each of the p dimensions, ||`spaceM2`||<||`spaceM1`|| at each dimension.
       * The Gauss quadrature nodes over 'spaceM2' can be the abscissa of different types of 
         polynomials based on the `distType` and the gPCE rule. 
       * A tensor-product grid from the GQ nodes on `spaceM2` is created

    args:    
      `qM1`: List of length p 
         List of samples on `spaceM1`; qM1=[qM1_1,qM1_2,...,qM1_p] where `qM1_i` is a 1D numpy 
         array of size nM1_i, for i=1,2,...,p. 
      `fValM1`: numpy p-D array of shape (nM1_1,nM1_2,...,nM1_p).
         Response values at `qM1`
      `spaceM1`: List of length p.
         =[spaceM1_1,spaceM1_2,...,spaceM1_p] where spaceM1_i is a list of two elements, 
         specifying the admissible range of qM1_i, for i=1,2,...,p.
      `nM2` List of length p,
         Containing the number of Gauss quadrature samples `qM2` in each parameter dimension, 
         nM2=[nM2_1,nM2_2,...,nM2_p]
      `spaceM2`: List of length p.
         =[spaceM2_1,spaceM2_2,...,spaceM2_p] where spaceM2_i is a list of two elements, 
         specifying the admissible range of qM2_i, for i=1,2,...,p.
      `distType`: List of length p with string members
         The i-th element specifies the distribution type of the i-th parameter according to 
         the gPCE rule.

    Returns:
      `qM2`: List of length p 
         List of samples on `spaceM2`; qM2=[qM2_1,qM2_2,...,qM2_p] where `qM2_i` is a 1D numpy 
         array of size nM2_i, for i=1,2,...,p. 
      `xiM2`: numpy array of shape (nM2_1*nM2_2*...*nM2_p,p)   
         Tensor-product grid of Gauss-quadrature nodes on the mapped space of `spaceM2`
      `fValM2`: 1D numpy array of size (nM1_1*nM1_2*...*nM1_p).
         Interpolated response values at `xiM2`
    """
    #(1) Check the inputs
    ndim=len(spaceM1)   
    if (ndim!=len(spaceM2) or ndim!=len(qM1)):
       raise ValueError('SpaceM1 and SpaceM2 should have the same dimension, p.')
    for idim in range(ndim):
       d1=spaceM1[idim][1]-spaceM1[idim][0]
       d2=spaceM2[idim][1]-spaceM2[idim][0]
       if (d2>d1):
          print("Wrong parameter range in direction ",ldim+1) 
          raise ValueError("||spaceM2|| should be smaller than ||spaceM1||.") 
    #(2) Construct the Gauss-quadrature stochastic samples for model2
    qM2=[]
    xiM2=[]
    for i in range(ndim):
        xi_,w=pce.gqPtsWts(nM2[i],distType[i])   
        qM2_=pce.mapFromUnit(xi_,spaceM2[i])
        qM2.append(qM2_)
        xiM2.append(xi_)
    if (ndim==1): 
       qM2=qM2[0]
       xiM2=xiM2[0]
    elif (ndim>1):
       xiM2=reshaper.vecs2grid(xiM2) 
    #(3) Use lagrange interpolation to find values at q2, given fVal1 at q1
    if ndim==1:
       fVal2Interp=lagInt(fNodes=fValM1,qNodes=[qM1[0]],qTest=[qM2]).val
    elif (ndim>1):
       fVal2Interp_=lagInt(fNodes=fValM1,qNodes=qM1,qTest=qM2,liDict={'testRule':'tensorProd'}).val
       nM2_=fVal2Interp_.size
       fVal2Interp=fVal2Interp_.reshape(nM2_,order='F')
    return qM2,xiM2,fVal2Interp
#
