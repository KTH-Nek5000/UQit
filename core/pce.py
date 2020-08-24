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
import plot2d
import writeUQ
import reshaper
import linAlg
import sampling
#
#
class pce:
   """
   Construction of non-inrusive generalized Polynomial Chaos Expansion (PCE)

   Parameters
   ----------

   Attributes
   ----------

   Methods
   -------


   """
   def __init__(self,fVal,xi,qInfo,pceDict,nQList=[]):
       self.fVal=fVal
       self.xi=xi
       self.nQList=nQList
       self.qInfo=qInfo
       self.pceDict=pceDict
       self.info()
       self.pceDict_corrector()
       self.cnstrct()

   def info(self):
       obligKeyList=['p','distType','sampleType','pceSolveMethod'] #obligatory keys in pceDict
       optKeyList=['truncMethod','LMax'] #optional keys in pceDict
       self.obligKeyList=obligKeyList
       self.optKeyList=optKeyList
       self.LMax_def=10   #default value of LMax (in case it is not provided)

   def pceDict_corrector(self):
       R"""
        Check and Correct pceDict for PCE to ensure consistency.
        * For 'GQ' samples+'TP' truncation method: either 'Projection' or 'Regression' can be used
        * For all combination of sample points and truncation, 'Projection' can be used to compute PCE coefficients
       """
       for key_ in self.obligKeyList:
           if key_ not in self.pceDict:
              raise KeyError("%s is missing from pceDict." %key_)
       if self.pceDict['p']>1:   
          if len(self.pceDict['distType'])!=self.pceDict['p']:
             raise ValueError("'p' is not equal to the number of %s in pceDict." %key_)

       if len(self.qInfo)!=self.pceDict['p']:
          raise ValueError("'p' is not equal to the length of 'qInfo'.")

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
                if len(self.nQList)!=self.pceDict['p']:
                   raise ValueError("For 'truncMethod':'TP', nQList of the length p=%d is needed."%self.pceDict['p']) 
             if self.pceDict['pceSolveMethod']=='Regression' and self.pceDict['truncMethod']!='TP':
                if 'LMax' not in self.pceDict:
                   print("WARNING in pceDict: 'LMax' should be set when Total-Order method is used.")
                   print("Here 'LMax' is set to default value %d" %self.LMax_def)
                   self.pceDict.update({'LMax':self.LMax_def})
       #Values associated to pceDict's obligatory keys            
       self.p=self.pceDict['p']                   
       self.pceSolveMethod=self.pceDict['pceSolveMethod']                   
       self.distType=self.pceDict['distType']                   
       self.sampleType=self.pceDict['sampleType']                   
       if 'LMax' in self.pceDict.keys():
          self.LMax=self.pceDict['LMax'] 
       if self.p>1:
          self.truncMethod=self.pceDict['truncMethod'] 

   @classmethod
   def mapToUnit(self,x_,xBound_):
      R"""
      Linearly map x\in[xBound] to \xi\in[-1,1]
      x can be either scalar or a vector
      """
      xi_=(2.*(x_-xBound_[0])/(xBound_[1]-xBound_[0])-1.)
      return xi_

   @classmethod
   def mapFromUnit(self,xi_,xBound_):
      R"""
      Linearly map \xi\in[-1,1] to x\in[xBound]
      x can be either scalar or a vector
      """
      xi = np.array(xi, copy=False, ndmin=1)
      x_=(0.5*(xi_+1.0)*(xBound_[1]-xBound_[0])+xBound_[0])
      return x_

   @classmethod
   def gqPtsWts(self,n_,type_):
      """
      Gauss quadrature nodes and weights associated to distribution type type_    
      (Based on the gPCE rule)
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
      Map \xi\in\Gamma to q \in Q, where \Gamma is the space corresponding to standard gPCE
        * xAux=xBound if distType='Unif', \xi\in[-1,1]
        * xAux=[m,v] if disType='Norm' where x~N(m,v), \xi\in[-\infty,\infty]
      """
#      xi = np.array(xi, copy=False, ndmin=1)
      if distType_=='Unif':
         x_=mapFromUnit(xi_,xAux_)
      elif distType_=='Norm':
         x_=xAux_[0]+xAux_[1]*xi_
      return x_    

   @classmethod
   def basis(self,n_,xi_,distType_):
      R"""
      Evaluate gPCE polynomial basis of order n at \xi\in\Gamma
      The standard polynomials are choosen based on the gPCE rules. 
      """
      if distType_=='Unif':
         v=np.polynomial.legendre.legval(xi_,[0]*n_+[1])
      elif distType_=='Norm':
         v=np.polynomial.hermite_e.hermeval(xi_,[0]*n_+[1])
      return v

   @classmethod
   def density(self,xi_,distType_):
      R"""
      Evaluate the PDF of the standard gPCE random variables with distribution
         type `distType_` at points \xi\in\Gamma (\Gamma: mapped space)
      """
      if distType_=='Unif':
         pdf_=0.5*np.ones(xi_.shape[-1])
      elif distType_=='Norm':
         pdf_=np.exp(-0.5*xi_**2.)/mt.sqrt(2*mt.pi)
      return pdf_

   def gqInteg_fac(self,distType_):
      """
      Multipliers for the GQ rule given the weights provided by numpy
      """
      if distType_=='Unif':
         fac_=0.5
      elif distType_=='Norm':
         fac_=1./mt.sqrt(2*mt.pi)
      return fac_
  
   def cnstrct(self):
       """
       Construct a PCE over a p-D parameter space
       """
       if self.p==1:
          self.cnstrct_1d() 
       elif self.p>1:
          self.cnstrct_pd()

   def cnstrct_1d(self):
      R""" 
      Construct a PCE over a 1D parameter space. (p=1) 

      Find :math:`{f_k}_{0:K}` in :math:`f(q)=\sum_k f_k psi_k(\xi)`, 
           where, K=truncation of the sum

      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
          Simulator's response values at `n` training samples
      `xi`: 1D numpy array 
          Training parameter samples over the mapped space.
          NOTE: Always had to be provided unless `'sampleType'`=`'GQ'` 
      `pceDict`: dictionary 
          containing different options for constructing the PCE with these keys:   
          `'p'`: int, dimension of the parameter
          `'distType'`: distribution type of the parameter based on gPCE rule
          `'pceSolveMethod'`: method of solving for PCE coefficients
             `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
             `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
             If under-determined, compressed sensing with L1/L2 regularization is automatically used.
          'sampleType'`: type of parameter samples at which observations are made
             ='GQ' (Gauss quadrature nodes)
             =' '  (other nodal sets, see `class trainSample` in `sampling.py`)
          `LMax`: int (Optional)
             Maximum order of PCE. `LMax` can be used only with `'pceSolveMethod':'Regression'`
             If `LMax` is not provided, it will be assumed to be equal to `n`.

      Attributes
      ----------
      `coefs`: Coefficients in the PCE, length=K
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]
      """
      if self.sampleType=='GQ' and self.pceSolveMethod=='Projection':
         self.cnstrct_GQ_1d()
      else:   #Regression method
         if 'LMax' in self.pceDict.keys():
            self.LMax=self.pceDict['LMax']
         else:
            self.LMax=len(self.fVal)
            self.pceDict['LMax']=self.LMax
            print("...... No 'LMax' existed, so 'LMax=nQ'")
         self.cnstrct_nonGQ_1d()

   def cnstrct_GQ_1d(self):
      R""" 
      Construct a PCE over a 1D parameter space using Projection method with Gauss-quadrature nodes.

      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
              Simulator's response values at `n` training samples
      `'distType'`: distribution type of the parameter based on gPCE rule

      Attributes
      ----------
      `coefs`: Coefficients in the PCE, length =K
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]       
      """
      nQ=len(self.fVal) #number of quadratures (collocation samples)
      xi,w=self.gqPtsWts(nQ,self.distType)
      K=nQ  #upper bound of sum in PCE
      #Find the coefficients in the expansion
      fCoef=np.zeros(nQ)
      sum2=[]
      fac_=self.gqInteg_fac(self.distType)
      for k in range(K):  #k-th coeff in PCE
          psi_k=self.basis(k,xi,self.distType)
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
      R""" 
      Construct a PCE over a 1D parameter space using Regression method 
          with arbitrary truncaton K=LMax to compute the PCE coefficients for arbitrary
          chosen parameter samples.

      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
              Simulator's response values at `n` training samples
      `'distType'`: distribution type of the parameter based on gPCE rule
      `xi`: 1D numpy array 
          Training parameter samples over the mapped space.
      `LMax`: int 
          Maximum order of PCE. `LMax` is required since `'pceSolveMethod':'Regression'`.

      Attributes
      ----------
      `coefs`: Coefficients in the PCE, length =K
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]       
      """
      nQ=len(self.fVal) #number of quadratures (collocation samples)
      K=self.LMax      #truncation in the PCE
      print('...... Number of terms in PCE, K= ',K)
      nData=len(self.fVal)   #number of observations
      print('...... Number of Data point, n= ',nData)
      #(2) Find the coefficients in the expansion:Only Regression method can be used. 
      #    Also we need to compute gamma_k (=sum2) required to estimate the variance by PCE.
      #    For this we create an auxiliary Gauss-Quadrature grid to compute intgerals
      A=np.zeros((nData,K))    #Matrix of known coeffcient for regression to compute PCE coeffcients
      sum2=[]
      xi_aux,w_aux=self.gqPtsWts(K+1,self.distType)  #auxiliary GQ rule for computing gamma_k
      fac_=self.gqInteg_fac(self.distType)
      for k in range(K):
          psi_aux=self.basis(k,xi_aux,self.distType)
          for j in range(nData):
              A[j,k]=self.basis(k,self.xi[j],self.distType)
          sum2.append(np.sum((psi_aux[:K+1])**2.*w_aux[:K+1]*fac_))
      #Find the PCE coeffs by Linear Regression 
      fCoef=linAlg.myLinearRegress(A,self.fVal)   #This can be over-, under-, or uniquely- determined systetm.
      #(3) Find the mean and variance of f(q) as estimated by PCE
      fMean=fCoef[0]
      fVar=np.sum(fCoef[1:]**2.*sum2[1:])
      self.coefs=fCoef
      self.fMean=fMean
      self.fVar=fVar

   def cnstrct_pd(self):
      R""" 
      Construct a PCE over a p-D parameter space, where p>1. 
      Find :math:`{f_k}_{0:K}` in :math:`f(q)=\sum_k f_k psi_k(\xi)`, 
           where, K=truncation of the sum
                  :math:`psi_k(\xi)=\psi_{k_1}(\xi_1)\psi_{k_2}(\xi_2)\cdots\psi_{k_p}(\xi_p)`
                 
      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
              Simulator's response values at `n` training samples
      `xi`: a 2D numpy array of shpae [n,p]
          Training parameter samples over the mapped space.
          NOTE: Always had to be provided unless `'sampleType'`=`'GQ'` 
      `pceDict`: dictionary 
          containing different options for constructing the PCE with these keys:             
          `'p'`: int, dimension of the parameter
          `'distType'`: distribution type of the parameter based on gPCE rule
          `'truncMethod'`: method of truncating PCE
              ='TP' (tensor product method)         
              ='TO' (total order method)
          `'pceSolveMethod'`: method of solving for PCE coefficients
             `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
             `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
               If under-determined, compressed sensing with L1/L2 regularization is automatically used.
          'sampleType'`: type of parameter samples at which observations are made
             ='GQ' (Gauss quadrature nodes)
             =' '  (other nodal sets, see `class trainSample` in `sampling.py`)
          `LMax`: int (Optional)
             Maximum order of PCE. `LMax` can be used only with `'pceSolveMethod':'Regression'`
             If `LMax` is not provided, it will be assumed to be equal to `n`.
       `nQList`: if 'truncMethod':'TP', then nQList=[nQ1,nQ2,...,nQp], 
                 where nQi: number of samples in i-th direction
                 if 'truncMethod':'TO': nQlist=[] (default)
      
      Attributes
      ----------
      `coefs`: Coefficients in the PCE, a 1d numpy array of size K
      `kSet`:  Index set, list (size K) of p-D lists [[k1,1,k2,1,...kp,1],...,[k1,K,k2,K,..,kp,K]]
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]
      """
      if self.sampleType=='GQ' and self.truncMethod=='TP':   #'GQ'+'TP': use either 'Projection' or 'Regression'
         self.cnstrct_GQTP_pd()
      else:                  #Any other type of samples (structured/unstructured): 'Regression' 
         self.cnstrct_nonGQTP_pd()

   def cnstrct_GQTP_pd(self):
      R"""
      Construct a PCE over a pD parameter space, for the following settings:
         * Type of parameter samples: 'GQ' (Gauss-Quadrature nodes)
         * Method of truncating PCE: 'TP' (Tnsor-product)
         * Method of solving for PCE coefficients: Projection or Regression

      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
              Simulator's response values at `n` training samples
      `pceDict`: dictionary 
          containing different options for constructing the PCE with these keys:   
          `'p'`: int, dimension of the parameter
          `'distType'`: distribution type of the parameter based on gPCE rule
          `'truncMethod'`: method of truncating PCE
              ='TP' (tensor product method)         
          `'pceSolveMethod'`: method of solving for PCE coefficients
             `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
             `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
               If under-determined, compressed sensing with L1/L2 regularization is automatically used.
          'sampleType'`: type of parameter samples at which observations are made
             ='GQ' (Gauss quadrature nodes)
       `nQList`: nQList=[nQ1,nQ2,...,nQp], 
                 where nQi: number of samples in i-th direction
              
      Attributes
      ----------
      `coefs`: Coefficients in the PCE, a 1d numpy array of size K
      `kSet`:  Index set, list (size K) of p-D lists [[k1,1,k2,1,...kp,1],...,[k1,K,k2,K,..,kp,K]]
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]
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
          fac.append(self.gqInteg_fac(distType[i]))
      print('...... Number of terms in PCE, K= ',K)
      nData=len(self.fVal)   #number of observations
      print('...... Number of Data point, n= ',nData)
      if K!=nData:
         raise ValueError("K=%d is not equal to nData=%d"%(K,nData)) 
         print('ERROR in pce_2d_GQTP_cnstrct(): ')
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
#
   def cnstrct_nonGQTP_pd(self):
      R"""
      Construct a PCE over a pD parameter space, for the following settings:
         * Method of truncating PCE: Total Order or Tensor Product
         * Method of solving for PCE coefficients: ONLY Regression
         * Use this method for any combination of 'sampleType' and
           'truncMethod' but 'GQ'+'TP'

      Parameters
      ----------
      `fVal`: 1D numpy array of size `n`
              Simulator's response values at `n` training samples
      `xi`: a 2D numpy array of shpae [n,p]
          Training parameter samples over the mapped space.
          NOTE: Always had to be provided unless `'sampleType'`=`'GQ'` 
      `pceDict`: dictionary 
          containing different options for constructing the PCE with these keys:   
          `'p'`: int, dimension of the parameter
          `'distType'`: distribution type of the parameter based on gPCE rule
          `'truncMethod'`: method of truncating PCE
             ='TP' (tensor product method)
             ='TO' (total order method)
          `'pceSolveMethod'`: method of solving for PCE coefficients
             `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
             `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
               If under-determined, compressed sensing with L1/L2 regularization is automatically used.
          'sampleType'`: type of parameter samples at which observations are made
             ='GQ' (Gauss quadrature nodes)
             =' '  (other nodal sets, see `class trainSample` in `sampling.py`)
       `nQList`: if 'truncMethod':'TP', then nQList=[nQ1,nQ2,...,nQp], 
                 where nQi: number of samples in i-th direction
                 if 'truncMethod':'TO': nQlist=[] (default)
      Attributes
      ----------
      `coefs`: Coefficients in the PCE, a 1d numpy array of size K
      `kSet`:  Index set, list (size K) of p-D lists [[k1,1,k2,1,...kp,1],...,[k1,K,k2,K,..,kp,K]]
      `fMean`: PCE estimation for E[f(q)]
      `fVar`:  PCE estimation for V[f(q)]
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
          fac.append(self.gqInteg_fac(distType[i]))
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
      #    Also we need to compute gamma_k (=sum2) required to estimate the variance by PCE.
      #    For this we create an auxiliary Gauss-Quadrature grid to compute intgerals
      A=np.zeros((nData,K))    #Matrix of known coeffcient for regression to compute PCE coeffcients
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
      fCoef=linAlg.myLinearRegress(A,self.fVal)   #This can be over-, under-, or uniquely- determined systetm.
      #(3) Find the mean and variance of f(q) as estimated by PCE
      fMean=fCoef[0]
      fVar=0.0
      for k in range(1,K):
          fVar+=fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
      self.coefs=fCoef
      self.fMean=fMean
      self.fVar=fVar
      self.kSet=kSet
#      
#
class pceEval:
   R"""
   Evaluation of a PCE at test samples taken from a parameter space
   """
   def __init__(self,coefs,xi,distType,kSet=[],convPltOpts=[]):
      self.coefs=coefs
      self.xi=xi
      self.distType=distType
      self.kSet=kSet
      self.convPltOpts=convPltOpts
      self.info()
      self.eval()
   
   def info(self):
       if len(self.kSet)==0:
          p=1
       else:
          p=len(self.distType)
       self.p=p   #param dimension
       K=len(self.coefs)
       self.K=K   #PCE truncation 
          
   def eval(self):    
      if self.p==1:
         self.eval_1d() 
      elif self.p>1:
         self.eval_pd() 

   def eval_1d(self):
      R""" 
      Evaluate a PCE over a 1D parameter space at a set of test points \xi 
         taken from \Gamma, the mapped parameter space.

      Parameters
      ----------
      `coefs`: 1D numpy array of length K
               PCE coefficients
      `xi`: 1D numpy array of size nTest
          Test parameter samples taken from the mapped space
      `distType`: string
          distribution type of the parameter based on the gPCE rule
          'Unif', 'Norm'

      Attribute   
      ---------
      `pceVal`: response values predicted (interpolated) by the PCE at `xi` test samples
      """
#    xi = np.array(xi, copy=False, ndmin=1)
      fpce=[]
      for i in range(self.xi.size):
          sum1=0.0
          for k in range(self.K):
              sum1+=self.coefs[k]*pce.basis(k,self.xi[i],self.distType)
          fpce.append(sum1)
      self.pceVal=np.asarray(fpce)

   def eval_pd(self):
      R""" 
      Evaluate a PCE built over a pD (p>1) parameter space at a set of test samples taken from the
      parameter mapped space. 

      Parameters
      ----------
      `coefs`: 1D numpy array of length K
               PCE coefficients
      `xi`: A list of length p
            containing the test points in each dimension of the mapped pD
                parameter space. xi=[xi_1,xi_2,..,xi_pp], where 
                xi_k is a 1d numpy array containing the test samples from the mapped space of the
                k-th parameter.
                Always a tensor product grid of test samples is constructed.
      `distType`: string
          distribution type of the parameter based on the gPCE rule
          'Unif', 'Norm'
      `kSet`: List of length K of lists of length p
          kSet=[[k1,1,k2,1,...,kp,1],[k1,2,k2,2,...,kp,2],...,[k1,K,k2,K,...,kp,K]] is 
          produced based on the truncation scheme employed when constructing the PCE.
          
      Attribute
      ---------
      `pceVal`: response values predicted (interpolated) by the PCE at `xi` test samples
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

   def convPlot(self):
      R"""
      Plot convergence of the PCE terms. 
      The indicator ::math::`||fk*Psi_k||/||f0*Psi_0||` is plotted versus 
      :math:`|k|=\sum_{i=1}^p k_i`          

      Parameters
      ----------
      `coefs`: 1D numpy array of length K
               PCE coefficients
      `kSet`: 
         if p>1: List of length K of lists of length p
            kSet=[[k1,1,k2,1,...,kp,1],[k1,2,k2,2,...,kp,2],...,[k1,K,k2,K,...,kp,K]] is 
            produced based on the truncation scheme employed when constructing the PCE.
            kSet: Index set of PCE, kSet=[[k1,k2,...,kp],...], if empty: 1d param space
         elif p==1:
            kset=[]
      `distType`: string
          distribution type of the parameter based on the gPCE rule
          'Unif', 'Norm'
      `convPltOpts`: (optional) options to save the figure, 
          keys: 'figDir', 'figName'           
      """
      K=self.K
      kSet_=[]
      if self.p==1:
         distType_=[self.distType] 
         for i in range(K):
             kSet_.append([i])
      else:
         kSet_=self.kSet 
         distType_=self.distType
      #magnitude of indices
      kMag=[]
      for i in range(K):
          kMag.append(sum(kSet_[i]))
      #compute norm of the PCE bases
      xi_=np.linspace(-1,1,1000)
      termNorm=[]
      for ik in range(K):   #over PCE terms
          PsiNorm=1.0
          for ip in range(self.p):   #over parameter dimension 
              k_=kSet_[ik][ip]
              psi_k_=pce.basis(k_,xi_,distType_[ip])
              PsiNorm*=np.linalg.norm(psi_k_,2)
              if distType_[ip] not in ['Unif','Norm']:
                 raise ValueError('...... ERROR in PCE_coef_conv_plot(): distribution %s is not available!'%distType_[ip])
          termNorm.append(abs(self.coefs[ik])*PsiNorm)
      termNorm0=termNorm[0]
      #plot
      plt.figure(figsize=(10,5))
      plt.semilogy(kMag,termNorm/termNorm0,'ob',fillstyle='none')
      plt.xlabel(r'$|\mathbf{k}|$',fontsize=18)
      plt.ylabel(r'$|\hat{f}_\mathbf{k}|\, ||\Psi_{\mathbf{k}(\mathbf{\xi})}||_2/|\hat{f}_0|$',fontsize=18)
      plt.xticks(ticks=kMag,fontsize=17)
      plt.yticks(fontsize=17)
      plt.grid(alpha=0.3)
      if self.convPltOpts:
         if 'ylim' in self.convPltOpts:
             plt.ylim(self.convPltOpts['ylim'])
         fig = plt.gcf()
         DPI = fig.get_dpi()
         fig.set_size_inches(800/float(DPI),400/float(DPI))
         figDir=self.convpltOpts['figDir']
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
    Test PCE for 1d uncertain parameter 
    """
    #--- settings -------------------------
    #Parameter settings
    distType='Unif'   #distribution type of the parameter
    if distType=='Unif':
       qInfo=[-2,4.0]   #parameter range only if 'Unif'
       fType='type1'    #Type of test exact model function
    elif distType=='Norm':
       qInfo=[0.1,0.6]   #[m,v] for 'Norm' q~N(m,v^2)
       fType='type2'    #Type of test exact model function
    n=16   #number of training samples
    nTest=200   #number of test sample sin the parameter space
    #PCE Options
    sampleType='GQ'    #'GQ'=Gauss Quadrature nodes
                       #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points 
                                #'Projection': only for GQ
    LMax_=20   #(Only needed for Regresson method), =K: truncation (num of terms) in PCE                               #(LMax will be over written by nSamples if it is provided for 'GQ'+'Projection')
               #NOTE: LMAX>=nSamples
    #--------------------------------------
    #(0) Make the pceDict
    pceDict={'p':1,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,'LMax':LMax_,
             'distType':distType}
    #
    #(1) Generate training data
    samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType,qInfo=qInfo,nSamp=n)
    q=samps.q
    xi=samps.xi
    qBound=samps.qBound
    f=analyticTestFuncs.fEx1D(q,fType,qInfo)   #function value at the parameter samples           
    #
    #(2) Compute the exact moments (as the reference data)
    fMean_ex,fVar_ex=analyticTestFuncs.fEx1D_moments(qInfo,fType)
    #
    #(3) Construct the PCE
    pce_=pce(fVal=f,xi=xi,qInfo=[qInfo],pceDict=pceDict)
    fMean=pce_.fMean  #mean, var estimated by the PCE and PCE coefficients
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    #
    #(4) Compare moments: exact vs. PCE estimations
    print(writeUQ.printRepeated('-',70))
    print('-------------- Exact -------- PCE --------- Error % ')
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_ex,fMean,(fMean-fMean_ex)/fMean_ex*100.))
    print('Var  of f(q) = %g\t%g\t%g' %(fVar_ex,fVar,(fVar-fVar_ex)/fVar_ex*100.))
    print(writeUQ.printRepeated('-',70))
    #
    #(5) Evaluate the PCE at test samples
    # Test samples
    testSamps=sampling.testSample('unifSpaced',GQdistType=distType,qInfo=qInfo,qBound=qBound,nSamp=nTest)
    qTest=testSamps.q
    xiTest=testSamps.xi
    fTest=analyticTestFuncs.fEx1D(qTest,fType,qInfo)   #exact response at test samples
    #Prediction by PCE at test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType)
    fPCE=pcePred_.pceVal
    #
    #(6) PLots
    # Plot convergence of the PCE
    pcePred_.convPlot()

    # Plot the exact and PCE response surface
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
    Test PCE for 2 uncertain parameters 
    """
    #---- SETTINGS------------
    distType=['Unif','Norm']   #distribution type of the parameters
    qInfo=[[-2,3],   #parameters info
           [-2,0.5]] 
    nQ=[7,9]   #number of collocation smaples of param1,param2: only for 'TP', otherwise =[]
    nTest=[121,120]   #number of test points in parameter spaces
    #PCE Options
    truncMethod='TP'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'     #'GQ'=Gauss Quadrature nodes
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
       fVal=analyticTestFuncs.fEx2D(q[0],q[1],fType,'tensorProd')  
       xiGrid=reshaper.vecs2grid(xi)
    elif sampleType=='LHS':
        if distType==['Unif']*p:
           qBound=qInfo
           xi=sampling.LHS_sampling(nQ[0]*nQ[1],[[-1,1]]*p)
           for i in range(p):
               q.append(pce.mapFromUnit(xi[:,i],qBound[i]))       
           fVal=analyticTestFuncs.fEx2D(q[0],q[1],fType,'pair')  
           xiGrid=xi
        else:  
           print("LHS works only when all q have 'Unif' distribution.") 
    #Make the pceDict       
    pceDict={'p':2,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    #Construct the PCE
    pce_=pce(fVal=fVal,xi=xiGrid,qInfo=qInfo,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet
    #Make predictions at test points in the parameter space
    qTest=[]
    xiTest=[]
    for i in range(p):
        testSamps=sampling.testSample('unifSpaced',GQdistType=distType[i],qInfo=qInfo[i],qBound=qBound[i],nSamp=nTest[i])
        qTest_=testSamps.q
        xiTest_=testSamps.xi
        qTest.append(qTest_)
        xiTest.append(xiTest_)
    fTest=analyticTestFuncs.fEx2D(qTest[0],qTest[1],fType,'tensorProd')
    #Evaluate PCE at the test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
    fPCE=pcePred_.pceVal

    #plot convergence of the PCE
    pcePred_.convPlot()

    #Create 2D grid from the test samples and plot the contours of response surface over it
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F')
    fErrorGrid=(abs(fTestGrid-fPCE))         
    #2d grid from the sampled parameters
    if sampleType=='LHS':
        q1Grid=q[0]
        q2Grid=q[1]
    else:
        q1Grid,q2Grid=plot2d.plot2D_grid(q[0],q[1])
    #plot 2d contours
    plt.figure(figsize=(21,8));
    plt.subplot(1,3,1)
    ax=plt.gca()
    CS1 = plt.contour(qTest[0],qTest[1],fTestGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='r',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.title('Exact Response')
    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(qTest[0],qTest[1],fPCE.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='r',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.title('Surrogate Response')
    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(qTest[0],qTest[1],fErrorGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2')
    plt.plot(q1Grid,q2Grid,'o',color='r',markersize=7)
    plt.title('|Exact-Surrogate|')
    plt.show()
#     
def pce_3d_test():
    """
    Test PCE for 3 uncertain parameters
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
    fVal=analyticTestFuncs.fEx3D(q[0],q[1],q[2],'Ishigami','tensorProd',funOpt)  
    #Make the pceDict
    pceDict={'p':3,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    #Construct the PCE   
    xiGrid=reshaper.vecs2grid(xi)
    pce_=pce(fVal=fVal,xi=xiGrid,qInfo=qInfo,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet

    #Exact moments of the Ishigami function
    m,v=analyticTestFuncs.ishigami_exactMoments(qBound[0],qBound[1],qBound[2],funOpt)
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
    fVal_test_ex=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',funOpt)  
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
