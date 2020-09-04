##########################################
#    Sobol Sensitivity Indices
##########################################
#------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#------------------------------------------
"""
  Assumptions:
    * parameters are independent from each other

  1. Extend to more than dual interactions 
  2. Move sobol_pd_test() to fExPD
  3. Add total Sobol to fEx funcs to validate the implementation
"""
#
import os
import sys
import numpy as np
from scipy.integrate import simps
sys.path.append(os.getenv("UQit"))
import analyticTestFuncs
import pce
import reshaper
#
class sobol:
    """
    Computing Sobol sensitivity indices for a p-D parameter (p>1).
    Assumptions:
      * Parameters are independent from each other
      * Up to second-order interactions are considered.
     
    Args:
      `q`: A list of length p
         q=[q1,q2,...,qp] where qi: 1D numpy array of size ni containing parameter samples
      'f': 1d numpy array of size (n1*n2*...*np) 
         Response values at samples `q`. The tensor product with ordering='F' (Fortran-like) is considered.
      'sampleType': List of length p of strings   
         The i-th value specifies the type of samples qi
      'distType': List of length p of strings   
         The i-th value specifies the type of distribution of the i-th parameter
    
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
    def __init__(self,q,f,sampleType,distType):
        self.q=q
        self.f=f
        self.sampleType=sampleType
        self.distType=distType
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
        self.L=[abs(max(self.q[i])-min(self.q[i])) for i in range(self.p)] #support
        self.distTypeList=['Unif','Norm']  #available distType
        #check disType & set integration method in the i-th dim
        integMethod=[]
        for i in range(self.p):
            if self.distType[i] not in self.distTypeList:
               print("Availble 'distType':",disTTypeList) 
               raise KeyError("Invalid distType for %d-D parameter." %self.p) 
            if self.sampleType[i]=='unifSpaced':
               integMethod.append('simps') 
            elif self.sampleType[i]=='GQ':
               integMethod.append('gq') 
        self.integMethod=integMethod       

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
          T`: scalar
            Inetgral value
       """
       T_=simps(g,x2,axis=1)
       T=simps(T_,x1)
       return T

    def dualInteractTerm(self,fij_,fi_,fj_):
        """
        Computes 2nd-order interaction terms in the HDMR decomposition based on 
        :math:`f_{ij}(qi,qj)=\int f(q)dq~{ij}-fi(qi)-fj(qj)-f0`
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
        L=self.L

        fi=[]
        for i in range(p):
            parIndexList=np.arange(p)  #Index of remaining parameters to integrate over          
            #permut list I=[]
            I_permute=self._permuteIndex(i)  #Permutation index list
            f_=self.f
            while I_permute.size>0:
                iInteg_=I_permute[0]
                iAxis_=np.where(parIndexList==iInteg_)[0][0]
                f_=simps(f_,q[iInteg_],axis=iAxis_)/L[iInteg_]
                parIndexList=np.delete(parIndexList,iAxis_)
                I_permute=np.delete(I_permute,0)
            fi.append(f_)    
        #compute the mean
        f0=simps(fi[0],q[0],axis=0)/L[0]
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
        L=self.L
        fij=[]
        for k in range(pairNum):
            indexList=np.arange(self.p)
            f_=self.f
            for l in range(compLen):
                iInteg_=compIndex[k][l]
                iAxis_=np.where(indexList==iInteg_)[0][0]
                if (self.p>2):
                   f_=simps(f_,q[iInteg_],axis=iAxis_)/L[iInteg_]
                else:
                   f_=self.p  
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
        fi=self.fi
        fij=self.fij
        L=self.L
        #1st-order terms, Di
        Di=[]
        for i in range(self.p):
            Di.append(simps(fi[i]**2.,q[i])/L[i])
        D=sum(Di)    
        #2nd-order terms, Dij
        Dij=[]
        SijName=[]
        #Main Sobol indices
        for k in range(len(fij)):
            i=self.dualInteractIndex[k][0]
            j=self.dualInteractIndex[k][1]
            Dij.append(self.doubleInteg(fij[k]**2.,q[i],q[j])/(L[i]*L[j]))
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
##########
##########
# TESTS
##########
#//////////////////////////
def sobol_2par_unif_test():
    """
      Test for sobol_unif() when we have 2 uncertain parameters q1, q2.
      Sobol indices are computed for f(q1,q2)=q1**2.+q1*q2 that is analyticTestFuncs.fEx2D('type3').
      Indices are computed from the following methods:
       * Method1: The Simpson numerical integration is used for the integrals in the definition of the indices (method of choise in myUQtoolbox).
       * Method2: First a PCE is constructed and then its predicitons at test points are used in Simpson integral of the Sobol indices.
       * Method3: Analytical expressions (see my notes)
    """
    #--------------------------
    #------- SETTINGS
    n=[101, 100]       #number of samples for q1 and q2, Method1
    qBound=[[-3,1],   #admissible range of parameters
            [-1,2]]
    nQpce=[5,6]      #number of GQ points for Method2
    #--------------------------
    fType='type3'    #type of analytical function
    p=len(n)
    distType=['Unif']*p
    #(1) Samples from parameters space
    q=[]
    for i in range(p):
        q.append(np.linspace(qBound[i][0],qBound[i][1],n[i]))

    #(2) Compute function value at the parameter samples
    fEx_=analyticTestFuncs.fEx2D(q[0],q[1],fType,'tensorProd')
    fEx=np.reshape(fEx_.val,n,'F')

    #(3) Compute Sobol indices direct numerical integration
    sampleType_=['unifSpaced']*2
    distType_=['Unif']*2
    sobol_=sobol(q,fEx,sampleType=sampleType_,distType=distType_)
    Si=sobol_.Si
    Sij=sobol_.Sij

    #(4) Construct a gPCE and then use the predictions of the gPCE in numerical integration for computing Sobol indices.
    #Generate observations at Gauss-Legendre points
    xi=[]
    qpce=[]
    for i in range(p):
        xi_,w_=pce.pce.gqPtsWts(nQpce[i],distType[i])
        qpce.append(pce.pce.mapFromUnit(xi_,qBound[i]))
        xi.append(xi_)
    fVal_pceCnstrct=analyticTestFuncs.fEx2D(qpce[0],qpce[1],fType,'tensorProd').val
    #Construct the gPCE
    xiGrid=reshaper.vecs2grid(xi)
    pceDict={'p':2,'sampleType':'GQ','truncMethod':'TP','pceSolveMethod':'Projection',
             'distType':distType}
    pce_=pce.pce(fVal=fVal_pceCnstrct,nQList=nQpce,xi=xiGrid,pceDict=pceDict)

    #Use gPCE to predict at test samples from parameter space
    qpceTest=[]
    xiTest=[]
    for i in range(p):
        qpceTest.append(np.linspace(qBound[i][0],qBound[i][1],n[i]))
        xiTest.append(pce.pce.mapToUnit(qpceTest[i],qBound[i]))
    fPCETest_=pce.pceEval(coefs=pce_.coefs,kSet=pce_.kSet,xi=xiTest,distType=distType)
    fPCETest=fPCETest_.pceVal
    #compute Sobol indices
    sobolPCE_=sobol(qpceTest,fPCETest,sampleType=sampleType_,distType=distType_)
    Si_pce=sobolPCE_.Si
    Sij_pce=sobolPCE_.Sij

    #(5) Exact Sobol indices (analytical expressions)
    if fType=='type3':
       fEx_.sobol(qBound)
       Si_ex=fEx_.Si
       Sij_ex=fEx_.Sij

    #(6) Write results on screen
    print(' > Indices by UQit:\n\t S1=%g, S2=%g, S12=%g' %(Si[0],Si[1],Sij[0]))
    print(' > gPCE+Numerical Integration:\n\t S1=%g, S2=%g, S12=%g' %(Si_pce[0],Si_pce[1],Sij_pce[0]))
    print(' > Analytical Reference:\n\t S1=%g, S2=%g, S12=%g' %(Si_ex[0],Si_ex[1],Sij_ex[0]))


#//////////////////////////
from math import pi
def sobol_3par_unif_test():
    """
      Test for sobol_unif() when we have 3 uncertain parameters q1, q2, q3.
      Sobol indices are computed for f(q1,q2,q3)=Ishigami that is analyticTestFuncs.fEx3D('Ishigami').
      First, we use Simpson numerical integration for the integrals in the definition of the indices (method of choice in myUQtoolbox). Then, these numerical values are validated by comparing them with the results of the analytical expressions.
    """
    #--------------------------
    #------- SETTINGS
    n=[100, 70, 80]       #number of samples for q1, q2, q3
    qBound=[[-pi,pi],      #admissible range of parameters
            [-pi,pi],
            [-pi,pi]]
    a=7   #parameters in Ishigami function
    b=0.1
    #--------------------------
    #(1) Samples from parameters space
    p=len(n)
    q=[]
    for i in range(p):
        q.append(np.linspace(qBound[i][0],qBound[i][1],n[i]))

    #(2) Compute function value at the parameter samples
    fEx_=analyticTestFuncs.fEx3D(q[0],q[1],q[2],'Ishigami','tensorProd',{'a':a,'b':b})
    fEx=np.reshape(fEx_.val,n,'F')

    #(3) Compute Sobol indices (method of choice in this library)
#    Si,Sij=sobol_unif(q,fEx)
    sampleType_=['unifSpaced']*3
    distType_=['Unif']*3
    sobol_=sobol(q,fEx,sampleType=sampleType_,distType=distType_)
    Si=sobol_.Si
    Sij=sobol_.Sij
    SijName=sobol_.SijName
    STi=sobol_.STi

    print('sobol_3par_unif_test(): Sobol Sensitivity Indices for fEx3D("Ishigami")')
    print(' > Main Indices by UQit : S1=%g, S2=%g, S3=%g' %(Si[0],Si[1],Si[2]))
    print(' >                        S12=%g, S13=%g, S23=%g' %(Sij[0],Sij[1],Sij[2]))
    print(' > Total                : ST1=%g, ST2=%g, ST3=%g' %(STi[0],STi[1],STi[2]))

    #(4) Exact Sobol indices (analytical expressions)
    fEx_.sobol(qBound)
    Si_ex=fEx_.Si
    Sij_ex=fEx_.Sij
    print(' > Main Analytical Reference: S1=%g, S2=%g, S3=%g' %(Si_ex[0],Si_ex[1],Si_ex[2]))
    print(' >                           S12=%g, S13=%g, S23=%g' %(Sij_ex[0],Sij_ex[1],Sij_ex[2]))
         
def sobol_pd_test():
    #---SETTINGS ---------------
    a=[0.5,0.2,1.2,0.4]
    p=len(a)
    qBound=[[0,1]]*p
    nSamp=[20,21,22,23]
    #---------------------------
    #Exact model functin
    q=[]
    for i in range(p):
        q_=np.linspace(qBound[i][0],qBound[i][1],nSamp[i])
        q.append(q_)
        fEx_=(abs(4*q_-2)+a[i])/(1+a[i])
        if i==0:
           fEx=fEx_
        else:
           fEx=np.tensordot(fEx,fEx_,0) 
    
    #Exact Sobol indices (Smith, p.336)
    Di=[]
    Dsum=1
    for i in range(p):
        Di.append(1/(3*(1+a[i])**2.))
        Dsum*=(1+Di[i])
    Dsum=-1+Dsum    
    Di=np.asarray(Di)
    Si=Di/Dsum
    Dij=[]
    for i in range(p):
        for j in range(p):
            if i!=j and i<j:
               Dij.append(Di[i]*Di[j])  
    Dij=np.asarray(Dij)
    Sij=Dij/Dsum
    print('Exact Sobol:',Si)
    print('Exact Sobol:',Sij)

    #Computed Sobol indices
    sampleType_=['unifSpaced']*p
    distType_=['Unif']*p
    sobol_=sobol(q,fEx,sampleType=sampleType_,distType=distType_)
    print('computed sobol:')
    print(sobol_.Si)
    print(sobol_.SijName)
    print(sobol_.Sij)
    print(sobol_.STi)
