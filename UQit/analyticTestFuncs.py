###################################################
# Analytical model functions to test implementation 
#   of different UQ techniques
###################################################
#--------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------
#TODO: Generalize Sobol of Ishigami to [ai,bi]
#--------------------------------------------
#
import os
import sys
import numpy as np
import math as mt
import UQit.sampling as sampling
#
#
class fEx1D:
    """
    Analytical test functions and their exact moments for 1D parameter

    Args:
      `q`: List or 1D numpy array of size n
         Samples of the parameter
      `typ`: string
         Function type, available: 'type1', 'type2':
           * 'type1' for `q~Uniform`
           * 'type2' for `q~Normal`
      `qInfo`: (optional) List (Required for the moments or if q is Gaussian)
           * `qInfo=[qMin,qMax]` if `q~U[qMin,qMax]`
           * `qInfo=[m,sdev]` if `q~N(m,sdev^2)` 

    Methods:
      `eval()`: 
         Evaluates f(q) at `q`
      `moments(qInfo)`: 
         Mean and variance of f(q)

    Returns:        
      `val`: 1D numpy array of size n,
         Values of f(q) at `q`
      `mean`: float
         E[f(q)] for `q`
      `var`: float
         V[f(q)] for `q`
    """
    def __init__(self,q,typ,qInfo=[]):
        self.q=q
        self.typ=typ
        self.qInfo=qInfo
        self._check()
        self.eval()

    def _check(self):
        if self.typ not in ['type1','type2']:
           raise ValueError("Invalid 'typ'. Choose 'type1' (for q~Uniform) or 'type2' (for q~Normal)")
    
    def eval(self):
        """
        Value of f(q) at q
        """
        z = np.array(self.q, copy=False, ndmin=1)
        if self.typ=='type1':
           val=(10.0+.7*np.sin(5.0*z)+3.*np.cos(z))
           val=np.array(val)
        elif self.typ=='type2':
           m=self.qInfo[0]
           val=np.cos(m*(z-m))
        self.val=val   

    def _mean1(self,q_):
        return (10.*q_-0.7*mt.cos(5.*q_)/5.0+3.*mt.sin(q_))

    def _var1(self,q_):
        tmp=100*q_+0.245*(q_-0.1*mt.sin(10*q_))+4.5*(q_+0.5*mt.sin(2*q_))
        return (tmp-2.8*mt.cos(5*q_)+60*mt.sin(q_)-2.1*(mt.cos(6*q_)/6.+0.25*mt.cos(4*q_)))

    def _mean2(self,q_):
        m=q_[0]
        v=q_[1]
        return(mt.exp(-0.5*(m*v)**2))      

    def _var2(self,q_):
        m=q_[0]
        v=q_[1]
        t_=(m*v)**2
        return(0.5*(1+mt.exp(-2*t_))-mt.exp(-t_))

    def moments(self,qInfo):
        """
        Analytical Mean and variance of f(q)
        """
        Q=qInfo
        if self.typ=='type1':
           fMean=(self._mean1(Q[1])-self._mean1(Q[0]))/(Q[1]-Q[0])                 
           fVar =(self._var1(Q[1])-self._var1(Q[0]))/(Q[1]-Q[0])-fMean**2.
        elif self.typ=='type2':
           fMean=self._mean2(Q)      
           fVar=self._var2(Q)
        self.mean=fMean   
        self.var=fVar
#
class fEx2D:
    """
    Analytical test functions for 2D parameter

    Args:  
      `q1`,`q2`: Two lists or 1D numpy arrays of size n1, n2, respectively.
         Samples of the parameters q1 and q2
      `typ`: string
         Function type, available: 'type1', 'type2', 'type3', 'Rosenbrock'.
      `method`: string
         Method for handling the multi-dimensionality: 'comp' or 'tensorProd'.
            * If 'comp' (component): n1 must be equal to n2 to make pairs of samples: q=(q1,q2).
            * If tensorProd' (tensor-product): size of `val` is n=n1*n2. 

    Methods:
      `eval()`: 
          Evaluates f(q) at (q1,q2)
      `sobol(qBound)`: 
          Sobol indices, available only for `typ`=='type3'.
          `qBound`: List of length 2, qBound=[qBound1,qBound2] where qBound_i is range of qi

    Returns:               
      `val`: 1D numpy array of size n
          Value of f(q) at q=(q1,q2)
            * If method=='comp': n=n1=n2
            * If method=='tensorProd': n=n1*n2
      `Si` : [S1,S2], 
          where `Si` is the main Sobol index with respect to the i-th parameter, i=1,2
      `STi` : [ST1,ST2], 
          where `Si` is the total Sobol index with respect to the i-th parameter, i=1,2
      `Sij`: [S12], 
          Dual interaction 
    """
    def __init__(self,q1,q2,typ,method):
        self.q1=q1
        self.q2=q2
        self.typ=typ
        self.method=method
        self._check()
        self.eval()

    def _check(self):
        method_valid=['comp','tensorProd']
        if self.method not in method_valid:
           raise ValueError("Invalid 'method'. Choose from ",method_valid) 
        typ_valid=['type1','type2','type3','Rosenbrock']
        if self.typ not in typ_valid:
           raise ValueError("Invalid 'typ'. Choose from ",typ_valid) 

    def _funVal(self,z1_,z2_,typ):
        if typ=='type1': # from: https://se.mathworks.com/help/symbolic/graphics.html?s_tid=CRUX_lftnav
           tmp1=3.*mt.exp(-(z2_+2)**2.-z1_**2) * (z1_-1.0)**2.
           tmp2=-(mt.exp(-(z1_+2)**2.-z1_**2))/3.
           tmp3=mt.exp(-(z1_**2.+z2_**2.))*(10.*z1_**3.-2.*z1_+10.*z2_**5.)
           tmp=tmp1+tmp2+tmp3
        elif typ=='type2':
           tmp1=3.*mt.exp(-(z2_+1)**2.-z1_**2) * (z1_-1.0)**2.
           tmp2=-1.*mt.exp(-(z2_-1.)**2.-z1_**2) * (z1_-1.)**2.
           tmp=tmp1+tmp2+0.001
        elif typ=='type3':   #simple enough for analytical derivation of Sobol indices
           tmp=z1_**2.+z1_*z2_
        elif typ=='Rosenbrock':
           tmp=100*(z2_-z1_**2.)**2.+(1-z1_)**2.
        return tmp
    
    def eval(self):
        """
        Evaluates f(q) at given q1, q2
        """
        z1 = np.array(self.q1, copy=False, ndmin=1)
        z2 = np.array(self.q2, copy=False, ndmin=1)
        n1=z1.shape[0]
        n2=z2.shape[0]
        f=[]
        if (self.method=='tensorProd'):
           for i2 in range(n2):
              z2_=z2[i2]
              for i1 in range(n1):
                 z1_=z1[i1]
                 tmp=self._funVal(z1_,z2_,self.typ)
                 f.append(tmp)
        elif (self.method=='comp'):
           if (n1!=n2):
              raise ValueError('Pairs of a paramater sample vector should have the same size')
           for i in range(n1):
              z1_=z1[i]
              z2_=z2[i]
              tmp=self._funVal(z1_,z2_,self.typ)
              f.append(tmp)
        self.val=np.asarray(f)

    def moments(self,distType,qInfo):
        """
        Mean and variance of f(q) estimated by the Monte-Carlo approach        
        (These can be used as reference values instead of the analytical values)

        Args:
           `distType`: List of length 2
               The i-th value (string) specifies the distribution type of the i-th parameter 
           `qInfo`: List of length 2
               Information about the parameter range or distribution.
                 * If `q` is Gaussian ('Norm' or 'normRand') => qInfo=[mean,sdev]
                 * Otherwise, qInfo=[min(q),max(q)]=admissible range of q
        
        Returns:
           `mean`: float
               Expected value of f(q) estimated by the Monte-Carlo method
           `var`: float
               Variance of f(q) estimated by the Monte-Carlo method
        """
        nMC=100000 #number of MC samples
        print('... Reference moments are calculated by the Monte-Carlo method with %d samples' %nMC)
        qMC=[]
        p=len(distType)
        if p!=2:
           raise ValueError("distType should have length 2") 
        for i in range(p):
            if distType[i]=='Unif':
               sampleType_='unifRand'
            elif distType[i]=='Norm':
               sampleType='normRand'
            else:
                raise ValueError("Invalid distType for parameter %d" %i)            
            samps=sampling.trainSample(sampleType=sampleType_,GQdistType=distType[i],
                  qInfo=qInfo[i],nSamp=nMC)
            qMC.append(samps.q)
        self.fVal_mc=fEx2D(qMC[0],qMC[1],self.typ,'comp').val
        self.mean=np.mean(self.fVal_mc)
        self.var=np.mean(self.fVal_mc**2.)-self.mean**2.

    def sobol(self,qBound):
        """
        Sobol sensitivity indices of f(q) with respect to q1 and q2
        
        Args:
          `qBound`: =[qBound1,qBound2] admissible range of q1, q2

        Returns:   
          `Si`:  =[S1,S2], 
             where `Si` is the main Sobol index wrt the i-th parameter, i=1,2
          `STi` : [ST1,ST2], 
             where `Si` is the total Sobol index wrt the i-th parameter, i=1,2
          `Sij`: =[S12], 
             dual interaction
        """
        if self.typ=='type3':
           #Exact sobol indices for f(q1,q2)=q1^2+q1*q2
           #->Take a,b according to eval() for 'type3'
           def _D1_Ex(a,b,q1):
              """
              a,b: terms in derived expression f1(q1)=q1^2+a*q1+b
              q1: sample for q1 at which the expression is evaluated
              """
              return(0.2*q1**5.+0.5*a*q1**4.+(a**2.+2.*b)/3.*q1**3.+a*b*q1**2.+b**2.*q1)
           def _D2_Ex(a,b,q2):
              """
              a,b: terms in derived expression f2(q2)=a*q2+b
              q2: sample for q2 at which the expression is evaluated
              """
              return((a**2./3.)*q2**3.+a*b*q2**2.+b**2.*q2)
           def _D12_Ex(a,b,c,q1Bound,q2Bound):
              """
              a,b,c: terms in derived expression f12(q1,q2)=q1*q2+a*q1+b*q2+c
              """
              q1_1=q1Bound[1]-q1Bound[0]
              q1_2=q1Bound[1]**2.-q1Bound[0]**2.
              q1_3=q1Bound[1]**3.-q1Bound[0]**3.
              q2_1=q2Bound[1]-q2Bound[0]
              q2_2=q2Bound[1]**2.-q2Bound[0]**2.
              q2_3=q2Bound[1]**3.-q2Bound[0]**3.
              return(q1_3*q2_3/9.+a/3.*q1_3*q2_2+b*q1_2*q2_3/3.+
                      0.5*(c+a*b)*q1_2*q2_2+a**2.*q1_3*q2_1/3.+b**2.*q1_1*q2_3/3.+
                      a*c*q1_2*q2_1+b*c*q1_1*q2_2+c**2.*q1_1*q2_1)
           #Variances in Sobol decomposition
           a1=qBound[0][0]
           b1=qBound[0][1]
           a2=qBound[1][0]
           b2=qBound[1][1]
           f0=(a1**2.+b1**2.+a1*b1)/3.+0.25*(a1+b1)*(a2+b2)
           a=(a2+b2)/2.0
           b=-f0
           D1_ex=(_D1_Ex(a,b,b1)-_D1_Ex(a,b,a1))/(b1-a1)
           a=(a1+b1)/2.0
           b=(a1**2.+a1*b1+b1**2.)/3.-f0
           D2_ex=(_D2_Ex(a,b,b2)-_D2_Ex(a,b,a2))/(b2-a2)
           a=-(a2+b2)/2.
           b=-(a1+b1)/2.
           c=f0-(a1**2.+a1*b1+b1**2.)/3.
           D12_ex=(_D12_Ex(a,b,c,qBound[0],qBound[1]))/((b1-a1)*(b2-a2))
           #Sensitivity indices
           D_ex=D1_ex+D2_ex+D12_ex
           Si=[D1_ex/D_ex,D2_ex/D_ex]
           Sij=[D12_ex/D_ex]
           STi=[Si[0]+Sij[0],Si[1]+Sij[0]]
           self.Si=Si
           self.Sij=Sij
           self.STi=STi
        else:
           print("No Exact Sobol indices for 'typ' else than 'type3'") 
           self.Si=[]
           self.Sij=[]
           self.STi=[]
#
class fEx3D:
    """
    Analytical test functions for 3D parameter

    Args:
      `q1`, `q2`, `q3`: Three lists or 1D numpy arrays of size n1, n2, n3, respectively
         Samples of the parameters `q1`, `q2` and `q3`
      `typ`: string
         Function type, available: 'Ishigami'
      `method`: string
         Method for handling the multi-dimensionality: 'comp' or 'tensorProd'
           * If 'comp' (component): n1 must be equal to n2. Pair of samples: q=(q1,q2,q3)
           * If 'tensorProd' (tensor-product): size of `val` is n=n1*n2*n3. 
      `opts`: function fixed parameters
         If 'Ishigami': opts=['a':a_val,'b':b_val]

    Methods:
      `eval()`: 
          Evaluates f(q) at (q1,q2,q3)
      `moments(qInfo)`: 
          Analytical values of mean and variance of f(q)
      `sobol(qBound)`: 
          Analytical Sobol indices
            
    Returns:
      `val`: 1D numpy array of size n
         Values of f(q) at q
           * If 'comp': n=n1=n2=n3
           * If 'tensorProd': n=n1*n2*n3
    """
    def __init__(self,q1,q2,q3,typ,method,opts):
        self.q1=q1
        self.q2=q2
        self.q3=q3
        self.typ=typ
        self.method=method
        self.opts=opts
        self._check()
        self.eval()

    def _check(self):
        method_valid=['comp','tensorProd']
        if self.method not in method_valid:
           raise ValueError("Invalid 'method'. Choose from ",method_valid) 
        typ_valid=['Ishigami']
        if self.typ not in typ_valid:
           raise ValueError("Invalid 'typ'. Choose from ",typ_valid) 

    def _funVal(self,z1_,z2_,z3_,typ,opts):
        if typ=='Ishigami': # Ishigami Function
           a=opts['a']
           b=opts['b']
           tmp=mt.sin(z1_)+a*(mt.sin(z2_))**2.+b*z3_**4.*mt.sin(z1_)
        return tmp

    def eval(self):
        z1 = np.array(self.q1, copy=False, ndmin=1)
        z2 = np.array(self.q2, copy=False, ndmin=1)
        z3 = np.array(self.q3, copy=False, ndmin=1)
        n1=z1.shape[0]
        n2=z2.shape[0]
        n3=z3.shape[0]
        f=[]
        if (self.method=='tensorProd'):
           for i3 in range(n3):
               z3_=z3[i3]
               for i2 in range(n2):
                   z2_=z2[i2]
                   for i1 in range(n1):
                       z1_=z1[i1]
                       tmp=self._funVal(z1_,z2_,z3_,self.typ,self.opts)
                       f.append(tmp)
        elif (self.method=='comp'):
           if (n1!=n2 or n1!=n3 or n2!=n3):
               raise ValueError("q1,q2,q3 should have the same length for 'method':'comp'")
           for i in range(n1):
              z1_=z1[i]
              z2_=z2[i]
              z3_=z3[i]
              tmp=self._funVal(z1_,z2_,z3_,self.typ,self.opts)
              f.append(tmp)
        self.val=np.asarray(f)

    def moments(self,qInfo):
        """ 
        Analytical mean and variance of f(q)

        Args:
          `qInfo`: List of length 3
             qInfo=[qBound_1,qBoun_2,qBound_3] where qBound_i: admissible range of the i-th parameter

        Returns:     
          `mean`: 
              Expected value of f(q)   
          `var`: 
              Variance of f(q)   
        """
        if self.typ=='Ishigami':
           #Analytical values of mean and variance of Ishigami function f(q1,q2,q3)
           #Assuming q1~U[a11,a12], q3~U[q21,q22], q3~U[a31,a32], 
           #   and q1,q2,q3 are mutually independent
           a1=qInfo[0]
           a2=qInfo[1]
           a3=qInfo[2]
           a1_=a1[1]-a1[0] #par range
           a2_=a2[1]-a2[0]
           a3_=a3[1]-a3[0]
           m=-a2_*a3_*(mt.cos(a1[1])-mt.cos(a1[0]))+0.5*self.opts['a']*a1_*a3_*(a2_-0.5*(mt.sin(2.*a2[1])-
              mt.sin(2*a2[0])))-0.2*self.opts['b']*a2_*(a3[1]**5.-a3[0]**5.)*(mt.cos(a1[1])-mt.cos(a1[0]))
           m=(m/(a1_*a2_*a3_))  

           SIN2_1=(mt.sin(2*a1[1])-mt.sin(2*a1[0]))
           SIN2_2=(mt.sin(2*a2[1])-mt.sin(2*a2[0]))
           SIN4_2=(mt.sin(4*a2[1])-mt.sin(4*a2[0]))
           v1=0.5*(a1_-0.5*SIN2_1)*(a3_+self.opts['b']**2.*(a3[1]**9.-a3[0]**9.)/9.0 + 
              0.4*self.opts['b']*(a3[1]**5.-a3[0]**5.))*a2_
           v2=-self.opts['a']*(mt.cos(a1[1])-mt.cos(a1[0]))*(a2_-0.5*SIN2_2)*(a3_+0.2*self.opts['b']*
                   (a3[1]**5.-a3[0]**5.))
           v3=0.25*self.opts['a']**2.*a1_*a3_*(1.5*a2_-SIN2_2+0.125*SIN4_2)
           v=v1+v2+v3
           v=v/(a1_*a2_*a3_) - m**2.  
        self.mean=m
        self.var=v

    def sobol(self,qBound):    
        """
        Sobol sensitivity indices of f(q) with respect to q1, q2, and q3
        
        Args: 
          `qBound`: List of length 3
             =[qBound1,qBound2,qBound3] admissible range of q1, q2, q3

        Returns:
          `Si`:  =[S1,S2,S3]
             where Si is Sobol indices with respect to qi
          `Sij`: =[S12,S13,S23]
             dual interactions
        """
        pi=mt.pi
        iFac=0
        for i in range(3):
            if qBound[i][0]!=-pi or qBound[i][1]!=pi:
               iFac=1

        if iFac==1:
           raise ValueError("qBound should be [-pi,pi] in all 3-directions.")
        
        # Exact Sobol indices for Ishigami function for q_i\in[-pi,pi], i=1,2,3
        a=self.opts['a']
        b=self.opts['b']
        D1=b*pi**4./5.+b**2.*pi**8./50. + 0.5
        D2=a**2./8.0
        D3=0.0
        D13=b**2.*pi**8./50.0+7.*b**2.*pi**8./450
        D12=0.0
        D23=0.0
        D123=0.0

        D=D1+D2+D3+D12+D13+D23+D123
        Si=[D1/D,D2/D,D3/D]
        Sij=[D12/D,D13/D,D23/D]
        self.Si=Si
        self.Sij=Sij
        STi=[Si[0]+Sij[0]+Sij[1],Si[1]+Sij[0]+Sij[2],Si[2]+Sij[1]+Sij[2]]
        self.STi=STi
#
