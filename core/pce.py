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
sys.path.append(os.getenv("UQit"))
import analyticTestFuncs
import writeUQ
import reshaper
import linAlg
import sampling
#
#
class pce:
   R"""
   Constructings non-inrusive generalized Polynomial Chaos Expansion (PCE).
   The parameter space has dimension p.
   We have taken n samples from the p-D parameter space and evaluated the 
   simulator at each sample. 
   The samples can be taken in any different approach, it is enough the options
   are set correctly. 
   The general aim is to, estimate :math:`\hat{f}_k` in 
   
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
      `nQList`: (optional) List of length p, `nQList=[nQ1,nQ2,...,nQp]`, 
          where `nQi`: number of samples in i-th direction
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
            * `'truncMethod'`: string (optional, only for p>1) 
                Method of truncating the PCE
                 - `'TP'`: Tensor-Product method     
                 - `'TO'`: Total-Order method
            * `'LMax'`: int (optional)
                Maximum order of the PCE in each of the parameter dimensions. 
                 It is mandatory for p>1 and `'TuncMethod'=='TO'`
                  - `'LMax'` can be used only with `'pceSolveMethod':'Regression'`
                  - If p==1 and `'LMax'` is not provided, it will be assumed to be equal to n.
                  - If p>1 and `'LMax'` is not provided, it will be assumed to a default value. 

   Attributes:
     `coefs`: 1D numpy array of size K 
        Coefficients in the PCE, `numpy` array of size K     
     `fMean`: scalar
        PCE estimation for E[f(q)]
     `fVar`: scalar
        PCE estimation for V[f(q)]
     `kSet`: List (size K) of p-D lists, p>1
        Index set :math:`[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]`
         - If p==1: `kSet=[]`
   """
   def __init__(self,fVal,xi,pceDict,nQList=[]):
       self.fVal=fVal
       self.xi=xi
       self.nQList=nQList
       self.pceDict=pceDict
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
                   print("... Original 'Projection' method for PCE is replaced by 'Regression'.")
       else:
          if self.pceDict['p']>1:
             if self.pceDict['truncMethod']=='TO':
                if 'pceSolveMethod' not in self.pceDict or self.pceDict['pceSolveMethod']!='Regression':
                   self.pceDict['pceSolveMethod']='Regression'
                   print("... Original method for PCE is replaced by 'Regression'.")
             if self.pceDict['truncMethod']=='TP':
                if 'sampleType' not in self.pceDict or self.pceDict['sampleType']!='GQ':
                   self.pceDict['pceSolveMethod']='Regression'
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
         `xi_`: Mapped `x_`
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
         `x_`: Mapped `xi_` 
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
          `n_`: int, order of the gPCE polynomial
          `type_`: string, distribution of the random variable according to the gPCE rule

       Returns:
          `quads`: 1D numpy array of size `n_`, Gauss quadrature nodes
          `weights`: 1D numpy array of size `n_`, Gauss quadrature weights
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
            * `xAux`=`xBound` if `distType_`=='Unif', hence :math:`\Gamma=[-1,1]`
            * `xAux`=`[m,sdev]` if `disType_`=='Norm', where `x_~N(m,sdev^2)` and :math:`\Gamma=[-\infty,\infty]`
       
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
       The standard polynomials are choosen based on the gPCE rules.        

       Args:
         `n_`: int
            Order of the basis
         `xi_`: 1D numpy array of size m
            Points on the mapped space
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
            L2-norm of the gPCe basis of order `k_`
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
             print("...... No 'LMax' existed, so 'LMax=n='",self.LMax)
          self.cnstrct_nonGQ_1d()

   def cnstrct_GQ_1d(self):
       R""" 
       Constructs a PCE over a 1D parameter space using Projection method with Gauss-quadrature nodes.

       Args:       
         `fVal`: 1D numpy array of size `n`
            Simulator's response values at `n` training samples
         `pceDict['distType']`: List of length 1, [distType1], distType1: string
            Distribution type of the parameter based on the gPCE rule
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
           with arbitrary truncaton `K=LMax` to compute the PCE coefficients for arbitrarily
           chosen parameter samples.

       Args:       
          `fVal`: 1D numpy array of size `n`
             Simulator's response values at `n` training samples
          `pceDict['distType']`: List of length 1, [distType1], distType1: string
             Distribution type of the parameter based on the gPCE rule
          `xi`: 2D numpy array of size (n,1)
              Training parameter samples over the mapped space 
          `pceDict['LMax']`: int 
             Maximum order of the PCE. 
             `LMax` is required since `'pceSolveMethod':'Regression'`.
       """
       nQ=len(self.fVal) #number of quadratures (collocation samples)
       K=self.LMax      #truncation in the PCE
       distType_=self.distType[0]
       print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
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
       Constructs a PCE over a pD parameter space, for the following settings:
          * `'sampType':'GQ'` (Gauss-Quadrature nodes)
          * `'truncMethod': 'TP'` (Tensor-product)
          * `'pceSolveMethod':'Projection'` or 'Regression'
       """
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
       print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
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
       Constructs a PCE over a pD parameter space, for the following settings:
          * `'reuncMethod'`: `'TO'` or `'TP'`
          * `'pceSolveMethod'`: ONLY `'Regression'`
          * This method is used  for any combination of `'sampleType'` and `'truncMethod'` 
            but `'GQ'`+`'TP'`
       """
       p=self.p
       distType=self.distType
       xiGrid=self.xi
       print('... A gPCE for a %d-D parameter space is constructed.' %p)
       print('...... PCE truncation method: %s' %self.truncMethod)
       print('...... Method of computing PCE coefficients: %s' %self.pceSolveMethod)
       if self.truncMethod=='TO':
          LMax=self.LMax   #max order of polynomial in each direction
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
       print('...... Number of terms in PCE, K= ',K)
       nData=len(self.fVal)   #number of observations
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
   """
   Evaluation of a PCE at test samples taken from the parameter space.
   The parameter space has dimension p.
   The number of test samples is m. 

   Args:
     `coefs`: 1D numpy array of size K
         PCE coefficients
     `xi`: A list of length p
         of the shape `xi=[xi_1,xi_2,..,xi_p]`, where `xi_k` is a 1D numpy array containing 
         `m_k` test samples taken from the mapped space of the k-th parameter. 
         Always a tensor-product grid of the test samples is constructed over the p-D space, 
         therefore, `m=m_1*m_2*...m_p`.
     `distType`: List of length p of strings, 
         The i-th value specifies the distribution type of the i-th parameter (based on the gPCE rule)
     `kSet`: (optional, required only if p>1) List of length `K`
         The index set produced when constructing the PCE with a specified truncation scheme.          
         :math:`kSet=[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]`

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
              sum1+=self.coefs[k]*pce.basis(k,xi_[i],self.distType)
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
   """
   Compute and plot the convergence indicator of a PCE that is defined as,

   .. math:: 
      ||f_k*\Psi_k||/||f0*\Psi_0||` 

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
         :math:`kSet=[[k_{1,1},k_{2,1},...k_{p,1}],...,[k_{1,K},k_{2,K},..,k_{p,K}]]`
     `convPltOpts`: (optional) dict
         Containing the options to save the figure;
            * 'figDir': Path to the directory at which the figure is saved (if not exists, is created)
            * 'figName': Name of the figure       
      
   Attributes:
     `kMag`: List of K integers
        `=|k|`, sum of the PCE uni-directional indices
     `pceConvIndic`: 1D numpy array of size K
        The PCE convergence indicator
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
#
############
# TESTS
############
#
def pce_1d_test():
    """
    Test PCE for 1D uncertain parameter 
    """
    #--- settings -------------------------
    #Parameter settings
    distType='Norm'   #distribution type of the parameter
    if distType=='Unif':
       qInfo=[-2,4.0]   #parameter range only if 'Unif'
       fType='type1'    #Type of test exact model function
    elif distType=='Norm':
       qInfo=[.5,0.9]   #[m,v] for 'Norm' q~N(m,v^2)
       fType='type2'    #Type of test exact model function
    n=20   #number of training samples
    nTest=200   #number of test sample sin the parameter space
    #PCE Options
    sampleType='GQ'    #'GQ'=Gauss Quadrature nodes
                       #''= any other sample => only 'Regression' can be selected
                       # see trainSample class in sampling.py
    pceSolveMethod='Projection' #'Regression': for any combination of sample points 
                                #'Projection': only for GQ
    LMax_=10   #(Only needed for Regresson method), =K: truncation (num of terms) in PCE                               #(LMax will be over written by nSamples if it is provided for 'GQ'+'Projection')
               #NOTE: LMAX>=nSamples
    #--------------------------------------
    #(0) Make the pceDict
    pceDict={'p':1,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,'LMax':LMax_,
             'distType':[distType]}

    #(1) Generate training data
    samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType,qInfo=qInfo,nSamp=n)
    q=samps.q
    xi=samps.xi
    qBound=samps.qBound
    fEx=analyticTestFuncs.fEx1D(q,fType,qInfo)   
    f=fEx.val

    #(2) Compute the exact moments (as the reference data)
    fEx.moments(qInfo)
    fMean_ex=fEx.mean
    fVar_ex=fEx.var

    #(3) Construct the PCE
    pce_=pce(fVal=f,xi=xi[:,None],pceDict=pceDict)
    fMean=pce_.fMean  #mean, var estimated by the PCE and PCE coefficients
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    
    #(4) Compare moments: exact vs. PCE estimations
    print(writeUQ.printRepeated('-',70))
    print('-------------- Exact -------- PCE --------- Error % ')
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_ex,fMean,(fMean-fMean_ex)/fMean_ex*100.))
    print('Var  of f(q) = %g\t%g\t%g' %(fVar_ex,fVar,(fVar-fVar_ex)/fVar_ex*100.))
    print(writeUQ.printRepeated('-',70))
    
    #(5) Plots
    # Plot convergence of the PCE
    convPlot(coefs=pceCoefs,distType=distType)
    #
    #(6) Evaluate the PCE at test samples
    # Test samples
    testSamps=sampling.testSample('unifSpaced',GQdistType=distType,qInfo=qInfo,qBound=qBound,nSamp=nTest)
    qTest=testSamps.q
    xiTest=testSamps.xi
    fTest=analyticTestFuncs.fEx1D(qTest,fType,qInfo).val   #exact response at test samples
    #Prediction by PCE at test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=[xiTest],distType=distType)
    fPCE=pcePred_.pceVal
    
    #(7) Plot the exact and PCE response surface
    plt.figure(figsize=(12,5))
    ax=plt.gca()
    plt.plot(qTest,fTest,'-k',lw=2,label=r'Exact $f(q)$')
    plt.plot(q,f,'ob',label=sampleType+' Training Samples')
    plt.plot(qTest,fPCE,'-r',lw=2,label='PCE')
    plt.plot(qTest,fMean*np.ones(len(qTest)),'-b',label=r'$\mathbb{E}[f(q)]$')
    ax.fill_between(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),color='powderblue',alpha=0.4)
    plt.plot(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b',label=r'$\mathbb{E}[f(q)]\pm 95\%CI$')
    plt.plot(qTest,fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b')
    plt.title('Example of 1D PCE for random variable of type %s' %distType)
    plt.xlabel(r'$q$',fontsize=19)
    plt.ylabel(r'$f(q)$',fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(alpha=0.3)
    plt.legend(loc='best',fontsize=17)
    plt.show()
#    
def pce_2d_test():
    """
    Test PCE for 2D uncertain parameter
    """
    #---- SETTINGS------------
    distType=['Unif','Unif']   #distribution type of the parameters
    qInfo=[[-2,3],   #parameters info
           [-2,0.2]] 
    nQ=[13,11]   #number of collocation smaples of param1,param2: only for 'TP', otherwise =[]
    nTest=[121,120]   #number of test points in parameter spaces
    #PCE Options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'   #'GQ'=Gauss Quadrature nodes
                      #''= any other sample (see sampling.py, trainSample) => only 'Regression' can be selected
                      #'LHS': Latin Hypercube Sampling (only when all distType='Unif')
    fType='type1'#'type2' 'Rosenbrock'     #type of exact model response                
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=10   #max polynomial order in each parameter direction
    #------------------------
    p=len(distType)
    #Generate training data
    xi=[]
    q=[]
    qBound=[]
    if sampleType=='GQ':
       for i in range(p):
           samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType[i],qInfo=qInfo[i],nSamp=nQ[i])
           q.append(samps.q)
           xi.append(samps.xi)
           qBound.append(samps.qBound)
       fVal=analyticTestFuncs.fEx2D(q[0],q[1],fType,'tensorProd').val  
       xiGrid=reshaper.vecs2grid(xi)
    elif sampleType=='LHS':
        if distType==['Unif']*p:
           qBound=qInfo
           xi=sampling.LHS_sampling(nQ[0]*nQ[1],[[-1,1]]*p)
           for i in range(p):
               q.append(pce.mapFromUnit(xi[:,i],qBound[i]))       
           fVal=analyticTestFuncs.fEx2D(q[0],q[1],fType,'comp').val  
           xiGrid=xi
        else:  
           raise ValueError("LHS works only when all q have 'Unif' distribution.") 
    #Make the pceDict       
    pceDict={'p':2,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    #Construct the PCE
    pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet
    #plot convergence of the PCE
    convPlot(coefs=pceCoefs,distType=distType,kSet=kSet)
    #Make predictions at test points in the parameter space
    qTest=[]
    xiTest=[]
    for i in range(p):
        testSamps=sampling.testSample('unifSpaced',GQdistType=distType[i],qInfo=qInfo[i],qBound=qBound[i],nSamp=nTest[i])
        qTest_=testSamps.q
        xiTest_=testSamps.xi
        qTest.append(qTest_)
        xiTest.append(xiTest_)
    fTest=analyticTestFuncs.fEx2D(qTest[0],qTest[1],fType,'tensorProd').val

    #Evaluate PCE at the test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
    fPCE=pcePred_.pceVal

    #Create 2D grid from the test samples and plot the contours of response surface over it
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F')
    fErrorGrid=(abs(fTestGrid-fPCE))         
    #2d grid from the sampled parameters
    if sampleType=='LHS':
       qGrid=reshaper.vecsGlue(q[0],q[1])
    else:
       qGrid=reshaper.vecs2grid(q)
    #plot 2d contours
    plt.figure(figsize=(21,8));
    plt.subplot(1,3,1)
    ax=plt.gca()
    CS1 = plt.contour(qTest[0],qTest[1],fTestGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.title('Exact Response')
    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(qTest[0],qTest[1],fPCE.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.title('Surrogate Response')
    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(qTest[0],qTest[1],fErrorGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.title('|Exact-Surrogate|')
    plt.show()
#     
def pce_3d_test():
    """
    Test PCE for 3D uncertain parameter
    """
    #----- SETTINGS------------
    distType=['Unif','Unif','Unif']
    qInfo=[[-0.75,1.5],   #range of param1
             [-0.5,2.5],   #range of param2
             [ 1.0,3.0]]   #range of param3
    nQ=[6,5,4] #number of parameter samples in the 3 dimensions
    funOpt={'a':7,'b':0.1}   #parameters in Ishigami function
    #PCE options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'   #'GQ'=Gauss Quadrature nodes
                      #any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    nTest=[5,4,3]   #number of test samples for the 3 parameters                          
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=10   #max polynomial order in each parameter direction
    #--------------------
    p=len(distType)
    #Generate training data
    xi=[]
    q=[]
    qBound=[]
    for i in range(p):
        samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType[i],qInfo=qInfo[i],nSamp=nQ[i])
        xi.append(samps.xi)
        q.append(samps.q)
        qBound.append(samps.qBound)
    fEx=analyticTestFuncs.fEx3D(q[0],q[1],q[2],'Ishigami','tensorProd',funOpt)  
    fVal=fEx.val
    #Make the pceDict
    pceDict={'p':3,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    #Construct the PCE   
    xiGrid=reshaper.vecs2grid(xi)
    pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet
    #Convergence of the PCE terms
    convPlot(coefs=pceCoefs,distType=distType,kSet=kSet)

    #Exact moments of the Ishigami function
    fEx.moments(qInfo=qBound)
    m=fEx.mean
    v=fEx.var
    #Comapre PCE and exact moments
    print(writeUQ.printRepeated('-',50))
    print('\t\t Exact \t\t PCE')
    print('E[f]:  ',m,fMean)
    print('V[f]:  ',v,fVar)
    #Compare the PCE predictions at test points with the exact values of the model response
    qTest=[]
    xiTest=[]
    for i in range(p):
        testSamps=sampling.testSample('unifSpaced',GQdistType=distType[i],qInfo=qInfo[i],qBound=qBound[i],nSamp=nTest[i])
        qTest.append(testSamps.q)
        xiTest.append(testSamps.xi)
    fVal_test_ex=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',funOpt).val  
    #PCE prediction at test points
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
    fVal_test_pce=pcePred_.pceVal

    nTest_=np.prod(np.asarray(nTest))
    fVal_test_pce_=fVal_test_pce.reshape(nTest_,order='F')
    err=np.linalg.norm(fVal_test_pce_-fVal_test_ex)
    plt.figure(figsize=(10,4))
    plt.plot(fVal_test_pce_,'-ob',mfc='none',ms=5,label='Exact')
    plt.plot(fVal_test_ex,'-xr',ms=5,label='PCE')
    plt.xlabel('Index of test samples, k')
    plt.ylabel('Model response')
    plt.legend(loc='best')
    plt.grid(alpha=0.4)
    plt.show()
    print('||fEx(q)-fPCE(q)|| % = ',err*100)
