##########################################
#    Sobol Sensitivity Indices
##########################################
#-----------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#-----------------------------------------
"""
  * Parameters are assumed to be independent from each other
  * Parameters can have any arbitrary distribution. The discrete PDF should be imported to sobol().
  * The samples generated for each of the parameters must be UNIFORMLY-spaced.
  * Up to second-order interactions are currently implemented. 
"""
#
#
import os
import sys
import numpy as np
from scipy.integrate import simps
#
class sobol:
    """
    Computes Sobol sensitivity indices for a p-D parameter (p>1).

    Assumptions:
      * Parameters are independent from each other
      * Up to second-order interactions are considered.
     
    Args:
      `q`: A list of length p
         q=[q1,q2,...,qp] where qi: 1D numpy array of size ni containing uniformly-spaced parameter samples
      `f`: 1d numpy array of size (n1*n2*...*np) 
         Response values at samples `q`. The tensor product with ordering='F' (Fortran-like) is considered.
      `pdf`: List of length p of 1D numpy arrays
         The i-th array in the list contains the values of the PDF of q_i, where i=1,2,...,p
    
    Methods:
       compute():
          Computes the Sobol indices

    Attributes:
       `Si`: A 1D numpy array of size p
          First-order main Sobol indices
       `Sij`: A 1D numpy array
          Second-order Sobol indices
       'SijName': A list 
          Containing the name of the Sobol indices represented by `Sij`
       `STi`: A 1D numpy array of size p
          Total Sobol indices
    """
    def __init__(self,q,f,pdf):
        self.q=q
        self.f=f
        self.pdf=pdf
        self._info()
        self.comp()

    def _info(self):
        """
        Settings and Checking
        """
        self.p=len(self.q)
        if self.p<2:
           raise ValueError("For Sobol indices at Least p==2 is required.")
        self.n=[self.q[i].shape[0] for i in range(self.p)] 

    def _permuteIndex(self,i):
        """
        Make a permutation list for index `i` over index set {0,1,...,p-1}
        Example: for  p=4, 
           i=0: I=[1,2,3]
           i=2: I=[3,0,1]
        """
        indexList=np.arange(self.p)
        I1=np.where(indexList<i)
        I2=np.where(indexList>i)
        I=np.append(I2,I1)
        return I    

    def _dualInteractIndex(self):
        """
        Makes a list of pairs of indices required for dual interactions
         * The indices in each pair are always ascending and non-repeated.
        """
        pairIndex=[]   #Pairs of indices whose interactions to be found
        indexList=np.arange(self.p)
        while indexList.size>1:
            iHead=indexList[0]
            indexList=np.delete(indexList,0)
            for i in range(indexList.size):
                pairIndex.append([iHead,indexList[i]])
        compIndex=[]  #Complement to the pairs (The dropped indices on which integration is done)
        indexList=np.arange(self.p)
        for pair_ in pairIndex:
            comp_=[]
            for ind_ in indexList:
                if ind_ not in pair_:
                   comp_.append(ind_) 
            compIndex.append(comp_)       
        return pairIndex,compIndex        

    def doubleInteg(self,g,x1,x2):
       """
       Numerical double integration :math:`\int_{x2} \int_{x1} g(x1,x2) dx1 dx2`
       
       Args:
         `g`: numpy array of shape (n1,n2)
            Values of integrand over the grid of `x1`-`x2`
         `x1`,`x2`: 1D numpy arrays of sizes n1 and n2
            Integrated variables            

       Returns:
         `T`: scalar
            Integral value
       """
       T_=simps(g,x2,axis=1)
       T=simps(T_,x1)
       return T

    def dualInteractTerm(self,fij_,fi_,fj_):
        """
        Computes 2nd-order interaction terms in the HDMR decomposition based on 
        :math:`f_{ij}(q_i,q_j)=\int f(q)dq_{\sim{ij}}-f_i(q_i)-f_j(q_j)-f_0`
        """
        ni=fi_.size
        nj=fj_.size
        fij=np.zeros((ni,nj))
        f0=self.f0
        for i2 in range(nj):
            for i1 in range(ni):
                fij[i1,i2]=fij_[i1,i2]-fi_[i1]-fj_[i2]-f0
        return fij

    def hdmr_0_1(self):
        """
        Zeroth- and first-order HDMR decomposition
        """
        p=self.p
        q=self.q
        pdf=self.pdf

        fi=[]
        for i in range(p):
            parIndexList=np.arange(p)  #Index of remaining parameters to integrate over          
            #permut list I=[]
            I_permute=self._permuteIndex(i)  #Permutation index list
            f_=self.f
            while I_permute.size>0:
                iInteg_=I_permute[0]
                iAxis_=np.where(parIndexList==iInteg_)[0][0]
                #reshape the pdf for multiplying it with f_
                n_=[1]*f_.ndim
                n_[iAxis_]=pdf[iInteg_].shape[0]
                pdf_=pdf[iInteg_].reshape(n_)  

                f_=f_*pdf_
                f_=simps(f_,q[iInteg_],axis=iAxis_)
                parIndexList=np.delete(parIndexList,iAxis_)
                I_permute=np.delete(I_permute,0)
            fi.append(f_)    
        #compute the mean
        f0=simps(fi[0]*pdf[0],q[0],axis=0)
        fi=[fi[i]-f0 for i in range(p)]
        self.f0=f0
        self.fi=fi

    def hdmr_2(self):
        """
        2nd-order HDMR decomposition
        """
        pairIndex,compIndex=self._dualInteractIndex()
        self.dualInteractIndex=pairIndex
        pairNum=len(pairIndex)  #number of dual interaction pairs
        compLen=len(compIndex[0])  #number of indices in each complement set
        q=self.q
        pdf=self.pdf
        fij=[]
        for k in range(pairNum):
            indexList=np.arange(self.p)
            f_=self.f
            for l in range(compLen):
                iInteg_=compIndex[k][l]
                iAxis_=np.where(indexList==iInteg_)[0][0]
                if (self.p>2):
                   #reshape the pdf for multiplying it with f_
                   n_=[1]*f_.ndim
                   n_[iAxis_]=pdf[iInteg_].shape[0]
                   pdf_=pdf[iInteg_].reshape(n_)  
                   f_=f_*pdf_
                   f_=simps(f_,q[iInteg_],axis=iAxis_)
                else:
                   f_=self.f
                indexList=np.delete(indexList,iAxis_)
            i=pairIndex[k][0]
            j=pairIndex[k][1]
            fi_=self.fi[i]
            fj_=self.fi[j]
            fij_=self.dualInteractTerm(f_,fi_,fj_) 
            fij.append(fij_)
        self.fij=fij    

    def hdmr(self):        
        """
        HDMR decomposition
        """
        #Mean and 1st-order Sobol decomposition terms
        self.hdmr_0_1()
        #2nd-order Interactions
        self.hdmr_2()

    def partVariance(self):
        """
        Partial variances and Sobol indices
        """
        q=self.q
        pdf=self.pdf
        fi=self.fi
        fij=self.fij
        #1st-order terms, Di
        Di=[]
        for i in range(self.p):
            Di_=simps(fi[i]**2.*pdf[i],q[i])
            Di.append(Di_)
        D=sum(Di)    
        #2nd-order terms, Dij
        Dij=[]
        SijName=[]
        #Main Sobol indices
        for k in range(len(fij)):
            i=self.dualInteractIndex[k][0]
            j=self.dualInteractIndex[k][1]
            pdf_=pdf[i][:,None]*pdf[j]
            Dij.append(self.doubleInteg(np.multiply(fij[k]**2.,pdf_),q[i],q[j]))
            SijName.append('S'+str(i+1)+str(j+1))
        D+=sum(Dij)    
        self.Si=Di/D
        self.Sij=Dij/D
        self.SijName=SijName
        #Total Sobol indices
        Sij_sum=[]
        for l in range(self.p):
            sum_=0
            for k in range(len(fij)):
                i=self.dualInteractIndex[k][0]
                j=self.dualInteractIndex[k][1]
                if i==l or j==l:
                   sum_+=self.Sij[k]
            Sij_sum.append(sum_)      
        self.STi=self.Si+Sij_sum           

    def comp(self):
        """
        Computes Sobol indices based on HDMR
        """
        #HDMR (Sobol) decomposition of the model function
        self.hdmr()
        #Partial Variances & Sobol indices
        self.partVariance()
#
