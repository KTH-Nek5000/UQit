#############################################
# generalized Polynomial Chaos Expansion
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
#
import os
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
import UQit.reshaper as reshaper
import UQit.linAlg as linAlg
#
#
class pce:
   R"""
   Constructs non-intrusive generalized Polynomial Chaos Expansion (PCE).
   The parameter space has dimension p.
   We have taken n samples from the p-D parameter space and evaluated the 
   simulator at each sample. 
   The samples can be taken using any approach, it is only enough to set the options correctly.
   The general aim is to estimate :math:`\hat{f}_k` in 
   
   .. math::
      f(q)=\sum_{k=0}^K \hat{f}_k \psi_k(\xi)
      
   where
   
   .. math::
      \psi_k(\xi)=\psi_{k_1}(\xi_1)\psi_{k_2}(\xi_2)\cdots\psi_{k_p}(\xi_p)

   Args:
      `fVal`: 1D numpy array of size n
          Simulator's response values at n training samples.
      `xi`: 2D numpy array of shape (n,p)
          Training parameter samples over the mapped space.
          NOTE: Always have to be provided unless `'sampleType':'GQ'` .
      `nQList`: (optional) List of length p, 
         `nQList=[nQ1,nQ2,...,nQp]`, where `nQi`: number of samples in i-th direction
            * nQList=[] (default) if p==1 or p>1 and `'truncMethod':'TO'`
            * Required only if p>1 and  `'truncMethod':'TP'` 
      `pceDict`: dict
          Contains settings required for constructing the PCE. The keys are:
            * `'p'`: int
               Dimension of the parameter
            * `'distType'`: List of length p, 
               The i-th value specifies the distribution type of the i-th parameter (based on the gPCE rule)
            * `'sampleType'`: string 
               Type of parameter samples at which observations are made
                - `'GQ'` (Gauss quadrature nodes)
                - `' '`  (other nodal sets, see `class trainSample` in `sampling.py`)
            * `'pceSolveMethod'`: string
               Method of solving for the PCE coefficients
                - `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
                - `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
                  If under-determined, compressed sensing with L1/L2 regularization is automatically used.
            * `'truncMethod'`: string (mandatory only if p>1) 
                Method of truncating the PCE
                 - `'TP'`: Tensor-Product method     
                 - `'TO'`: Total-Order method
            * `'LMax'`: int (optional)
                Maximum order of the PCE in each of the parameter dimensions. 
                 It is mandatory for p>1 and `'TuncMethod'=='TO'`
                  - `'LMax'` can be used only with `'pceSolveMethod':'Regression'`
                  - If p==1 and `'LMax'` is not provided, it will be assumed to be equal to n.
                  - If p>1 and `'LMax'` is not provided, it will be assumed to a default value. 
      `verbose`: bool (optional)
          If True (default), info is printed about the PCE being constructed

   Attributes:
     `coefs`: 1D numpy array of size K 
        Coefficients in the PCE
     `fMean`: scalar
        PCE estimation for E[f(q)]
     `fVar`: scalar
        PCE estimation for V[f(q)]
     `kSet`: List (size K) of p-D lists, p>1
        Index set :math:`[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]`
        If p==1, then kSet=[]
   """
   def __init__(self,fVal,xi,pceDict,nQList=[],verbose=True):
       self.fVal=fVal
       self.xi=xi
       self.nQList=nQList
       self.pceDict=pceDict
       self.verbose=verbose
       self._info()
       self._pceDict_corrector()
       self.cnstrct()

   def _info(self):
       """
       Checks consistency of the inputs
       """
       obligKeyList=['p','distType','sampleType','pceSolveMethod'] #obligatory keys in pceDict
       optKeyList=['truncMethod','LMax'] #optional keys in pceDict
       self.obligKeyList=obligKeyList
       self.optKeyList=optKeyList
       self.LMax_def=10   #default value of LMax (if not provided)
       if self.fVal.ndim >1:
          raise ValueError("fVal should be a 1D numpy array of size n.") 
       self.availDist=['Unif','Norm'] #Currently available distributions

   def _pceDict_corrector(self):
       """
       Checks and corrects `pceDict` to ensure the consistency of the options.
          * For 'GQ' samples+'TP' truncation scheme: either 'Projection' or 'Regression' can be used
          * For all combination of samples and truncation schemes, 'Projection' can be used to compute PCE coefficients
       """
       for key_ in self.obligKeyList:
           if key_ not in self.pceDict:
              raise KeyError("%s is missing from pceDict." %key_)
       self.p=self.pceDict['p']                   
       self.distType=self.pceDict['distType']                   
       self.sampleType=self.pceDict['sampleType']                   
       self.pceSolveMethod=self.pceDict['pceSolveMethod']                   
       
       if len(self.distType)!=self.p:
          raise ValueError("Provide %d 'distType' for the random parameter" %self.p)
       if len(self.nQList)>0 and len(self.nQList)!=self.p:
          raise ValueError("For 'truncMethod':'TP', nQList of the length p=%d is needed."%self.pceDict['p'])
       if len(self.xi)>0: #if xi is provided
          if self.xi.ndim!=2:
             raise ValueError("Wrong dimension: xi must be a 2D numpy array.")     
          else:
             if self.xi.shape[1]!=self.p:
                raise ValueError("Second dimension of xi should be equal to p="%self.p)      
           
       if 'truncMethod' not in self.pceDict:
          if self.pceDict['p']!=1:
             raise KeyError("'truncMethod' is missing from pceDict.")
          else: #p==1
             if self.pceDict['pceSolveMethod']=='Projection':
                if self.pceDict['sampleType'] !='GQ':
                   self.pceDict['pceSolveMethod']='Regression'
                   if self.verbose:
                      print("... Original 'Projection' method for PCE is replaced by 'Regression'.")
       else:
          if self.pceDict['p']>1:
             if self.pceDict['truncMethod']=='TO':
                if 'pceSolveMethod' not in self.pceDict or self.pceDict['pceSolveMethod']!='Regression':
                   self.pceDict['pceSolveMethod']='Regression'
                   if self.verbose:
                      print("... Original method for PCE is replaced by 'Regression'.")
             if self.pceDict['truncMethod']=='TP':
                if 'sampleType' not in self.pceDict or self.pceDict['sampleType']!='GQ':
                   self.pceDict['pceSolveMethod']='Regression'
                   if self.verbose:
                      print("... Original method for PCE is replaced by 'Regression'.")
             if self.pceDict['pceSolveMethod']=='Regression' and self.pceDict['truncMethod']!='TP':
                if 'LMax' not in self.pceDict:
                   print("WARNING in pceDict: 'LMax' should be set when Total-Order method is used.")
                   print("Here 'LMax' is set to default value %d" %self.LMax_def)
                   self.pceDict.update({'LMax':self.LMax_def})
       #Values associated to pceDict's obligatory keys            
       if 'LMax' in self.pceDict.keys():
          self.LMax=self.pceDict['LMax'] 
       if self.p>1:
          self.truncMethod=self.pceDict['truncMethod'] 
       #Check the validity of the distType
       for distType_ in self.distType:   
          if distType_ not in self.availDist:
             raise ValueError("Invalid 'distType'! Availabe list: ",self.availDist) 

   @classmethod
   def mapToUnit(self,x_,xBound_):
       R"""
       Linearly maps `x_` in `xBound_` to `xi_` in `[-1,1]`

       Args:
         `x_`: Either a scalar or a 1D numpy array

         `xBound_`: A list of length 2 specifying the range of `x_`            

       Returns:
         `xi_`: Mapped value of `x_`
       """
       xi_=(2.*(x_-xBound_[0])/(xBound_[1]-xBound_[0])-1.)
       return xi_

   @classmethod
   def mapFromUnit(self,xi_,xBound_):
       R"""
       Linearly maps `xi_`  in `[-1,1]` to `x` in `xBound_`
     
       Args:
         `xi_`: Either a scalar or a 1D numpy array

         `xBound_`: A list of length 2 specifying the range of `x_`            

       Returns:
         `x_`: Mapped value of `xi_` 
       """
       xi_ = np.array(xi_, copy=False, ndmin=1)
       x_=(0.5*(xi_+1.0)*(xBound_[1]-xBound_[0])+xBound_[0])
       return x_

   @classmethod
   def gqPtsWts(self,n_,type_):
       R"""
       Gauss quadrature nodes and weights associated to distribution type `type_`
       based on the gPCE rule.

       Args:
          `n_`: int, 
             Order of the gPCE polynomial.
          `type_`: string, 
             Distribution of the random variable according to the gPCE rule

       Returns:
          `quads`: 1D numpy array of size `n_` 
             Gauss quadrature nodes
          `weights`: 1D numpy array of size `n_` 
             Gauss quadrature weights
       """
       if type_=='Unif':
          x_=np.polynomial.legendre.leggauss(n_)
       elif type_=='Norm':
          x_=np.polynomial.hermite_e.hermegauss(n_)
       quads=x_[0]
       weights=x_[1]
       return quads,weights

   @classmethod
   def map_xi2q(self,xi_,xAux_,distType_):
       R"""
       Maps `xi_` in :math:`\Gamma` to `x_` in :math:`\mathbb{Q}`, where :math:`\Gamma` is the 
       mapped space corresponding to the standard gPCE. 

       Args:
         `xi_`: 1D numpy array of size n
            Samples taken from the mapped parameter space
         `distType_`: string 
            Distribution type of the parameter
         `xAux`: List of length 2
            * xAux==xBound (admissible range) if distType_=='Unif', hence :math:`\Gamma=[-1,1]`
            * xAux==[m,sdev]  if disType_=='Norm', where x_~N(m,sdev^2) and :math:`\Gamma=[-\infty,\infty]`
       
       Returns:
         `x_`: 1D numpy array of size n
            Mapped parameter value in :math:`\mathbb{Q}`
       """
       if distType_=='Unif':
          x_=mapFromUnit(xi_,xAux_)
       elif distType_=='Norm':
          x_=xAux_[0]+xAux_[1]*xi_
       return x_    

   @classmethod
   def basis(self,n_,xi_,distType_):
       R"""
       Evaluates gPCE polynomial basis of order `n_` at `xi_` points taken from the mapped 
       space :math:`\Gamma`.
       The standard polynomials are chosen based on the gPCE rules.        

       Args:
         `n_`: int
            Order of the basis
         `xi_`: 1D numpy array of size m
            Points taken from the mapped space
         `distType_`: string
            Distribution type of the random parameter (based on the gPCE rule)
       
       Returns:
         `psi`: 1D numpy array of size m
            Values of the gPCE basis at `xi_`     
       """
       if distType_=='Unif':
          psi=np.polynomial.legendre.legval(xi_,[0]*n_+[1])
       elif distType_=='Norm':
          psi=np.polynomial.hermite_e.hermeval(xi_,[0]*n_+[1])
       return psi

   @classmethod
   def basisNorm(self,k_,distType_,nInteg=10000):
       """
       Evaluates the L2-norm of the gPCE polynomial basis of order `k_` at `nInteg` points taken from the 
       mapped space :math:`\Gamma`.

       Args:
         `k_`: int
            Order of the gPCE basis
         `distType_`: string
            Distribution type of the random parameter (according to the gPCE rule)
         `nInteg`: int (optional)  
            Number of points to evaluate the L2-norm integral

       Returns:
         `xi_`: scalar
            L2-norm of the gPCE basis of order `k_`
       """
       if distType_=='Unif':
          xi_=np.linspace(-1,1,nInteg)
       elif distType_=='Norm':
          xi_=np.linspace(-5,5,nInteg)          
       psi_k_=self.basis(k_,xi_,distType_)
       psiNorm=np.linalg.norm(psi_k_,2)
       return psiNorm

   @classmethod
   def density(self,xi_,distType_):
       R"""
       Evaluates the PDF of the standard gPCE random variables with distribution 
       type `distType_` at points `xi_` taken from the mapped space :math:`\Gamma`. 

       Args:
         `xi_`: 1D numpy array of size n
            Samples taken from the mapped space
         `distType_`: string
            Distribution type of the random parameter (according to the gPCE rule)

       Returns:
         `pdf_`: 1D numpy array of size n
            PDF of the random parameter (according to the gPCE rule)
       """
       if distType_=='Unif':
          pdf_=0.5*np.ones(xi_.shape[-1])
       elif distType_=='Norm':
          pdf_=np.exp(-0.5*xi_**2.)/mt.sqrt(2*mt.pi)
       return pdf_

   def _gqInteg_fac(self,distType_):
       """
       Multipliers for the Gauss-quadrature integration rule, when using the 
       weights provided by numpy.
       """
       if distType_=='Unif':
          fac_=0.5
       elif distType_=='Norm':
          fac_=1./mt.sqrt(2*mt.pi)
       return fac_
  
   def cnstrct(self):
       """
       Constructs a PCE over a p-D parameter space (p=1,2,3,...)
       """
       if self.p==1:
          self.cnstrct_1d() 
       elif self.p>1:
          self.cnstrct_pd()

   def cnstrct_1d(self):
       R""" 
       Constructs a PCE over a 1D parameter space, i.e. p=1.
       """
       if self.sampleType=='GQ' and self.pceSolveMethod=='Projection':
          self.cnstrct_GQ_1d()
       else:   #Regression method
          if 'LMax' in self.pceDict.keys():
             self.LMax=self.pceDict['LMax']
          else:
             self.LMax=len(self.fVal)
             self.pceDict['LMax']=self.LMax
             if self.verbose:
                print("...... No 'LMax' existed, so 'LMax=n='",self.LMax)
          self.cnstrct_nonGQ_1d()

   def cnstrct_GQ_1d(self):
       R""" 
       Constructs a PCE over a 1D parameter space using Projection method with Gauss-quadrature nodes.

       Args:       
         `fVal`: 1D numpy array of size `n`
            Simulator's response values at `n` training samples
         `pceDict['distType']`: List of length 1
            =[distType1], where distType1:string specifies the distribution type of the parameter 
            based on the gPCE rule
       """
       nQ=len(self.fVal) 
       distType_=self.distType[0]
       xi,w=self.gqPtsWts(nQ,distType_)
       K=nQ  
       #Find the coefficients in the expansion
       fCoef=np.zeros(nQ)
       sum2=[]
       fac_=self._gqInteg_fac(distType_)
       for k in range(K):  #k-th coeff in PCE
           psi_k=self.basis(k,xi,distType_)
           sum1=np.sum(self.fVal[:K]*psi_k[:K]*w[:K]*fac_)
           sum2_=np.sum((psi_k[:K])**2.*w[:K]*fac_)
           fCoef[k]=sum1/sum2_
           sum2.append(sum2_)
       #Estimate the mean and variance of f(q) 
       fMean=fCoef[0]
       fVar=np.sum(fCoef[1:]**2.*sum2[1:])
       self.coefs=fCoef
       self.fMean=fMean
       self.fVar=fVar

   def cnstrct_nonGQ_1d(self):
       """ 
       Constructs a PCE over a 1D parameter space using 'Regression' method 
       with arbitrary truncation `K=LMax` to compute the PCE coefficients for arbitrarily
       chosen parameter samples.

       Args:       
          `fVal`: 1D numpy array of size `n`
             Simulator's response values at `n` training samples
          `pceDict['distType']`: List of length 1, 
             =[distType1], where distType1:string specifies the distribution type of the parameter 
             based on the gPCE rule
          `xi`: 2D numpy array of size (n,1)
              Training parameter samples over the mapped space 
          `pceDict['LMax']`: int 
             Maximum order of the PCE. 
             `LMax` is required since `'pceSolveMethod'=='Regression'`.
       """
       nQ=len(self.fVal) #number of quadratures (collocation samples)
       K=self.LMax      #truncation in the PCE
       distType_=self.distType[0]
       if self.verbose:
          print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
       if self.verbose:
          print('...... Number of Data point, n= ',nData)
       #(2) Find the coefficients in the expansion:Only Regression method can be used. 
       A=np.zeros((nData,K))    
       sum2=[]
       xi_aux,w_aux=self.gqPtsWts(K+1,distType_)  #auxiliary GQ rule for computing gamma_k
       fac_=self._gqInteg_fac(distType_)
       for k in range(K):
           psi_aux=self.basis(k,xi_aux,distType_)
           for j in range(nData):
               A[j,k]=self.basis(k,self.xi[j],distType_)
           sum2.append(np.sum((psi_aux[:K+1])**2.*w_aux[:K+1]*fac_))
       #Find the PCE coeffs by Linear Regression 
       fCoef=linAlg.myLinearRegress(A,self.fVal) 
       #(3) Find the mean and variance of f(q) estimated by PCE
       fMean=fCoef[0]
       fVar=np.sum(fCoef[1:]**2.*sum2[1:])
       self.coefs=fCoef
       self.fMean=fMean
       self.fVar=fVar

   def cnstrct_pd(self):
       """ 
       Construct a PCE over a p-D parameter space, where p>1. 
       """
       if self.sampleType=='GQ' and self.truncMethod=='TP':   
            #'GQ'+'TP': use either 'Projection' or 'Regression'
          self.cnstrct_GQTP_pd()
       else: #'Regression'
          self.cnstrct_nonGQTP_pd()

   def cnstrct_GQTP_pd(self):
       R"""
       Constructs a PCE over a pD parameter space (p>1) using the following settings:
          * `'sampType':'GQ'` (Gauss-Quadrature nodes)
          * `'truncMethod': 'TP'` (Tensor-product)
          * `'pceSolveMethod':'Projection'` or 'Regression'
       """
       if self.verbose:
          print('... A gPCE for a %d-D parameter space is constructed.'%self.p)
          print('...... Samples in each direction are Gauss Quadrature nodes (User should check this!).')
          print('...... PCE truncation method: TP')
          print('...... Method of computing PCE coefficients: %s' %self.pceSolveMethod)
       distType=self.distType
       p=self.p
       #(1) Quadrature rule
       xi=[]
       w=[]
       fac=[]
       K=1
       for i in range(p):
           xi_,w_=self.gqPtsWts(self.nQList[i],distType[i])
           xi.append(xi_)
           w.append(w_)
           K*=self.nQList[i]
           fac.append(self._gqInteg_fac(distType[i]))
       if self.verbose:    
          print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
       if self.verbose:
          print('...... Number of Data point, n= ',nData)
       if K!=nData:
          raise ValueError("K=%d is not equal to nData=%d"%(K,nData)) 
       #(2) Index set
       kSet=[]    #index set for the constructed PCE
       kGlob=np.arange(K)   #Global index
       kLoc=kGlob.reshape(self.nQList,order='F')  #Local index
       for i in range(K):
           k_=np.where(kLoc==kGlob[i])
           kSet_=[]
           for j in range(p):
               kSet_.append(k_[j][0])
           kSet.append(kSet_)
       #(3) Find the coefficients in the expansion
       #By default, Projection method is used (assuming samples are Gauss-Quadrature points)
       fCoef=np.zeros(K)
       sum2=[]
       fVal_=self.fVal.reshape(self.nQList,order='F').T #global to local index
       for k in range(K):
           psi_k=[]
           for j in range(p):
               psi_k.append(self.basis(kSet[k][j],xi[j],distType[j]))
           sum1 =np.matmul(fVal_,(psi_k[0]*w[0]))*fac[0]
           sum2_=np.sum(psi_k[0]**2*w[0])*fac[0]
           for i in range(1,p):
               num_=(psi_k[i]*w[i])
               sum1=np.matmul(sum1,num_)*fac[i]
               sum2_*=np.sum(psi_k[i]**2*w[i])*fac[i]
           fCoef[k]=sum1/sum2_
           sum2.append(sum2_)
       #(3b) Compute fCoef via Regression
       if self.pceDict['pceSolveMethod']=='Regression':
          xiGrid=reshaper.vecs2grid(xi)
          A=np.zeros((nData,K))
          for k in range(K):
              aij_=self.basis(kSet[k][0],xiGrid[:,0],distType[0])
              for i in range(1,p):
                  aij_*=self.basis(kSet[k][i],xiGrid[:,i],distType[i])
              A[:,k]=aij_
          fCoef=linAlg.myLinearRegress(A,self.fVal)   #This is a uniquely determined system
       #(4) Find the mean and variance of f(q) as estimated by PCE
       fMean=fCoef[0]
       fVar=np.sum(fCoef[1:]**2.*sum2[1:])
       self.coefs=fCoef
       self.fMean=fMean
       self.fVar=fVar
       self.kSet=kSet

   def cnstrct_nonGQTP_pd(self):
       R"""
       Constructs a PCE over a pD parameter space (p>1), for the following settings:
          * `'truncMethod'`: `'TO'` or `'TP'`
          * `'pceSolveMethod'`: `'Regression'` (only allowed method)
          * This method is used  for any combination of `'sampleType'` and `'truncMethod'` 
            but `'GQ'`+`'TP'`
       """
       p=self.p
       distType=self.distType
       xiGrid=self.xi
       if self.verbose:
          print('... A gPCE for a %d-D parameter space is constructed.' %p)
          print('...... PCE truncation method: %s' %self.truncMethod)
          print('...... Method of computing PCE coefficients: %s' %self.pceSolveMethod)
       if self.truncMethod=='TO':
          LMax=self.LMax   #max order of polynomial in each direction
          if self.verbose:
             print('         with LMax=%d as the max polynomial order in each direction.' %LMax)
       if self.pceSolveMethod!='Regression':
          raise ValueError("only Regression method can be used for PCE with Total Order truncation method.") 
       #(1) Preliminaries
       #Number of terms in PCE
       if self.truncMethod=='TO':
          K=int(mt.factorial(LMax+p)/(mt.factorial(LMax)*mt.factorial(p)))
          Nmax=(LMax+1)**p
          kOrderList=[LMax+1]*p
       elif self.truncMethod=='TP':   #'TP' needs nQList
          K=np.prod(np.asarray(self.nQList))
          Nmax=K
          kOrderList=self.nQList
       # Quadrature rule only to compute \gamma_k
       xiAux=[]
       wAux=[]
       fac=[]
       for i in range(p):
           xi_,w_=self.gqPtsWts(self.nQList[i],distType[i])
           xiAux.append(xi_)
           wAux.append(w_)
           fac.append(self._gqInteg_fac(distType[i]))
       # Index set
       kSet=[]    #index set for the constructed PCE
       kGlob=np.arange(Nmax)   #Global index
       kLoc=kGlob.reshape(kOrderList,order='F')  #Local index
       for i in range(Nmax):
           k_=np.where(kLoc==kGlob[i])
           kSet_=[]
           for j in range(p):
               kSet_.append(k_[j][0])
           if (self.truncMethod=='TO' and sum(kSet_)<=LMax) or self.truncMethod=='TP':
              kSet.append(kSet_)
       if self.verbose:       
          print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
       if self.verbose:
          print('...... Number of Data point, n= ',nData)
       #(2) Find the coefficients in the expansion:Only Regression method can be used.
       A=np.zeros((nData,K))    
       sum2=[]
       for k in range(K):
           psi_k_=self.basis(kSet[k][0],xiAux[0],distType[0])
           sum2_=np.sum(psi_k_**2*wAux[0])*fac[0]
           aij_=self.basis(kSet[k][0],xiGrid[:,0],distType[0])
           for i in range(1,p):
               aij_*=self.basis(kSet[k][i],xiGrid[:,i],distType[i])
               psi_k_=self.basis(kSet[k][i],xiAux[i],distType[i])
               sum2_*=np.sum(psi_k_**2*wAux[i])*fac[i]
           A[:,k]=aij_
           sum2.append(sum2_)
       #Find the PCE coeffs by Linear Regression
       fCoef=linAlg.myLinearRegress(A,self.fVal) 
       #(3) Find the mean and variance of f(q) as estimated by PCE
       fMean=fCoef[0]
       fVar=0.0
       for k in range(1,K):
           fVar+=fCoef[k]*fCoef[k]*sum2[k]   
       self.coefs=fCoef
       self.fMean=fMean
       self.fVar=fVar
       self.kSet=kSet
#
class pceEval:
   R"""
   Evaluates a constructed PCE at test samples taken from the parameter space.
   The parameter space has dimension p.
   The number of test samples is m. 

   Args:
     `coefs`: 1D numpy array of size K
         PCE coefficients
     `xi`: A list of length p
         `xi=[xi_1,xi_2,..,xi_p]`, where `xi_k` is a 1D numpy array containing 
         `m_k` test samples taken from the mapped space of the k-th parameter. 
         Always a tensor-product grid of the test samples is constructed over the p-D space, 
         therefore, `m=m_1*m_2*...*m_p`.
     `distType`: List of length p of strings, 
         The i-th value specifies the distribution type of the i-th parameter (based on the gPCE rule)
     `kSet`: (optional, required only if p>1) List of length `K`
         The index set produced when constructing the PCE with a specified truncation scheme.          
         :math:`kSet=[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]` with :math:`k_{i,j}` being integer

   Attribute:
     `pceVal`: 1D numpy array of size m,
         Response values predicted (interpolated) by the PCE at the test samples 
   """
   def __init__(self,coefs,xi,distType,kSet=[]):
      self.coefs=coefs
      self.xi=xi
      self.distType=distType
      self.kSet=kSet
      self._info()
      self.eval()
   
   def _info(self):
       p=len(self.distType)
       self.p=p   #param dimension
       K=len(self.coefs)
       self.K=K   #PCE truncation 
       self.availDist=['Unif','Norm'] #available distributions
       #Check the validity of the distType
       distType_=self.distType
       for distType__ in distType_:   
          if distType__ not in self.availDist:
             raise ValueError("Invalid 'distType'! Availabe list: ",self.availDist) 
          
   def eval(self):    
      if self.p==1:
         self.eval_1d() 
      elif self.p>1:
         self.eval_pd() 

   def eval_1d(self):
      """ 
      Evaluates a PCE over a 1D parameter space at a set of test samples xi 
      taken from the mapped parameter space.
      """
      fpce=[]
      xi_=self.xi[0]
      for i in range(xi_.size):
          sum1=0.0
          for k in range(self.K):
              sum1+=self.coefs[k]*pce.basis(k,xi_[i],self.distType[0])
          fpce.append(sum1)
      self.pceVal=np.asarray(fpce)

   def eval_pd(self):
      """ 
      Evaluates a PCE over a p-D (p>1) parameter space at a set of test samples xi 
      taken from the mapped parameter space.
      """
      p=self.p
      K=self.K
      for k in range(K):        
          a_=self.coefs[k]*pce.basis(self.kSet[k][0],self.xi[0],self.distType[0])
          for i in range(1,p):
              b_=pce.basis(self.kSet[k][i],self.xi[i],self.distType[i])
              a_=np.matmul(np.expand_dims(a_,axis=a_.ndim),b_[None,:])
          if k>0:
             A+=a_ 
          else:
             A=a_ 
      fpce=A    
      self.pceVal=fpce

class convPlot:
   R"""
   Computes and plots the convergence indicator of a PCE that is defined as,

   .. math:: 
      \vartheta_k = ||f_k \Psi_k||/|f_0|

   versus :math:`|k|=\sum_{i=1}^p k_i`.
   The dimension of the parameter space, p, is arbitrary. But for p>1, kSet have to 
   be provided. 

   Args:
     `coefs`: 1D numpy array of length K
         The PCE coefficients
     `distType`: List of length p of strings, 
         The i-th value specifies the distribution type of the i-th parameter (based on the gPCE rule)
     `kSet`: (optional, required only if p>1) List of length `K`
         The index set produced when constructing the PCE with a specified truncation scheme.          
         :math:`kSet=[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]` with :math:`k_{i,j}` being integer.
     `convPltOpts`: (optional) dict
         Containing the options to save the figure. It includes the following keys:
            * 'figDir': 
               Path to the directory at which the figure is saved (if not exists, is created)
            * 'figName': 
               Name of the figure       
      
   Attributes:
     `kMag`: List of K integers
        `=|k|`, sum of the PCE uni-directional indices
     `pceConvIndic`: 1D numpy array of size K
        The PCE convergence indicator

   Methods:
     `pceConv()`:
         Computes the convergence indicator 

     `pceConvPlot()`:
         Plots the PCE convergence indicator
   """
   def __init__(self,coefs,distType,kSet=[],convPltOpts=[]):
      self.coefs=coefs
      self.distType=distType
      self.kSet=kSet
      self.convPltOpts=convPltOpts
      self._get_info()
      self.pceConv()
      self.pceConvPlot()
      
   def _get_info(self):
       if len(self.kSet)==0:
          p=1
       else:
          p=len(self.distType)
       self.p=p   #param dimension
       K=len(self.coefs)
       self.K=K   #PCE truncation 
       self.availDist=['Unif','Norm'] #available distributions
       #Check the validity of the distType
       if self.p==1:
          distType_=[self.distType]
       else:
          distType_=self.distType
       for distType__ in distType_:   
          if distType__ not in self.availDist:
             raise ValueError("Invalid 'distType'! Availabe list: ",self.availDist) 

   def pceConv(self):
       """
       Computes the convergence indicator 
       """
       K=self.K
       if self.p==1:
          distType_=[self.distType] 
          kSet_=[[i] for i in range(K)]
       else:
          kSet_=self.kSet 
          distType_=self.distType
       #magnitude of the pce indices
       kMag=[sum(kSet_[i]) for i in range(K)]
       #compute norm of the PCE bases
       termNorm=[]
       for ik in range(K):   #over PCE terms
           PsiNorm=1.0
           for ip in range(self.p):   #over parameter dimension 
               k_=kSet_[ik][ip]
               PsiNorm*=pce.basisNorm(k_,distType_[ip])
           termNorm.append(abs(self.coefs[ik])*PsiNorm)
           termNorm0=termNorm[0]
       self.kMag=kMag    
       self.pceConvIndic=termNorm/termNorm0    

   def pceConvPlot(self):        
       """
       Plots the PCE convergence indicator
       """
       plt.figure(figsize=(10,5))
       plt.semilogy(self.kMag,self.pceConvIndic,'ob',fillstyle='none')
       plt.xlabel(r'$|\mathbf{k}|$',fontsize=18)
       plt.ylabel(r'$|\hat{f}_\mathbf{k}|\, ||\Psi_{\mathbf{k}(\mathbf{\xi})}||_2/|\hat{f}_0|$',fontsize=18)
       plt.xticks(ticks=self.kMag,fontsize=17)
       plt.yticks(fontsize=17)
       plt.grid(alpha=0.3)
       if self.convPltOpts:
          if 'ylim' in self.convPltOpts:
              plt.ylim(self.convPltOpts['ylim'])
          fig = plt.gcf()
          DPI = fig.get_dpi()
          fig.set_size_inches(800/float(DPI),400/float(DPI))
          figDir=self.convPltOpts['figDir']
          if not os.path.exists(figDir):
             os.makedirs(figDir)
          outName=self.convPltOpts['figName']
          plt.savefig(figDir+outName+'.pdf',bbox_inches='tight')
          plt.show()
       else:
          plt.show()
#          
