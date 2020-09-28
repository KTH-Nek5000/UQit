#################################################################
#            Sobol Sensitivity Indices
#################################################################
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------------
import os
import sys
import numpy as np
from scipy.integrate import simps
sys.path.append(os.getenv("UQit"))
import analyticTestFuncs
import pce
import reshaper

#/////////////////////////////
def doubleInteg(g,x1,x2):
    """
       Numerical double integral \int_{x2} \int_{x1} g(x1,x2) dx1 dx2
       g: 2d numpy array of size (n1,n2)
       x1,x2: 1d numpy arrays of sizes n1 and n2, resp.       
    """
    T_=simps(g,x2,axis=1)
    T=simps(T_,x1)
    return T

#/////////////////////////////////
def sobolDecomposCoefs_2par(Q,f):
    """
       Compute coefficients f0, f1(q1), f2(q2), f12(q1,q2) in the 2nd-order Sobol decomposition (HDMR)
       Q=[Q1|Q2] a list of two 1d numpy array Qi of size ni
       f=2d numpy array of size(n1,n2)
    """
    p=2
    n=[]
    L=[]
    for i in range(p):
        n.append(len(Q[i]))   
        L.append(abs(max(Q[i])-min(Q[i])))
    #Compute f1(q1), f2(q2) and q0
    fi=[]
    for i in range(p):
        I=p-i-1
        fi.append(simps(f,Q[I],axis=-I)/L[I])   #fi(qi)+q0
    print(fi[0].shape,L)    
    f0=simps(fi[0],Q[0])/L[0]          #q0=E(f(q1,q2))
    fi[:]=fi[:]-f0   #fi(qi)
    fij=np.zeros((n[0],n[1]))   #f12(q1,q2): interaction term
    for i2 in range(n[1]):
        for i1 in range(n[0]):
            fij[i1,i2]=f[i1,i2]-fi[0][i1]-fi[1][i2]-f0
    return f0,fi,fij,L

#////////////////////////////
def sobolDecomposCoefs_3par(Q,f):
    """
       Compute coefficients f0, fi(qi) (i=1,2,3), fij(qi,qj) (i,j=1,2,3) in the 2nd-order Sobol decomposition (HDMR)
       Q=[Q1|Q2|Q3] a list of three 1d numpy array Qi of size ni
       f=3d numpy array of size (n1,n2,n3) 
    """
    def interactionTerm(fij_,fi,fj,f0):
        """
           Compute 2nd-order interaction term in HDMR 
           f_{ij}(qi,qj)=\int f(q)dq~{ij}-fi(qi)-fj(qj)-f0
        """
        n1=fi.size
        n2=fj.size
        fij=np.zeros((n1,n2))   #f12(q1,q2): interaction term
        for i2 in range(n2):
            for i1 in range(n1):
                fij[i1,i2]=fij_[i1,i2]-fi[i1]-fj[i2]-f0
        return fij
    p=3
    n=[]
    L=[]
    for i in range(p):
        n.append(len(Q[i]))   
        L.append(abs(max(Q[i])-min(Q[i])))

    #Compute first order interactions fi(qi)=\int f(q1,q2,q3) dq~i, i=1,2,3 and q0
    print('f',f.shape)
    f1_=simps(f  ,Q[1],axis=1)/L[1]   #f1_=\int f  dq2  
    print('f1_',f1_.shape)
    f1 =simps(f1_,Q[2],axis=1)/L[2]   #f1 =\int f1_dq3
    print('f1',f1.shape)
    f2_=simps(f  ,Q[2],axis=2)/L[2]   #f2_=\int f  dq3
    print('f2_',f2_.shape)
    f2 =simps(f2_,Q[0],axis=0)/L[0]   #f2 =\int f2_dq1
    print('f2',f2.shape)
    f3_=simps(f  ,Q[0],axis=0)/L[0]   #f3_=\int f  dq1
    print('f3_',f3_.shape)
    f3 =simps(f3_,Q[1],axis=0)/L[1]   #f3 =\int f3_dq2
    print('f3',f3.shape)
    f0 =simps(f1,Q[0],axis=0)/L[0]
    f1=f1-f0
    f2=f2-f0
    f3=f3-f0      
    fi=[f1,f2,f3]
    #2nd-order Interaction terms 
    f12_=simps(f,Q[2],axis=2)/L[2]
    f12=interactionTerm(f12_,f1,f2,f0)
    f13_=simps(f,Q[1],axis=1)/L[1]
    f13=interactionTerm(f13_,f1,f3,f0)
    f23_=simps(f,Q[0],axis=0)/L[0]
    f23=interactionTerm(f23_,f2,f3,f0)
    fij=[f12,f13,f23]
    return f0,fi,fij,L

#////////////////////////////
def sobol_unif(Q,f):
    """
       Compute Sobol sensitivity indices for f(Q1,Q2,...,Qp), where q1,q2~Uniform. 
       Q=[Q1|Q2|...|Qp], list of p 1d numpy array
       Qi:numpy 1d array of size ni, i=1,2,...,p
       f: numpy p-d array: f(n1,n2,...,np)
    """
    p=len(Q)
    if p==2:
       #Sobol (HDMR) decomposition
       f0,fi,f12,L=sobolDecomposCoefs_2par(Q,f)       
       #1st-order variances
       Si=[]
       for i in range(p):
           Si.append(simps(fi[i]**2.,Q[i])/L[i])
       #2nd-order variance
       Sij=[]
       Sij.append(doubleInteg(f12**2.,Q[0],Q[1])/(L[0]*L[1]))
    elif (p==3):
       #Sobol (HDMR) decomposition
       f0,fi,fij,L=sobolDecomposCoefs_3par(Q,f)       
       #1st-order variances
       Si=[]
       for i in range(p):
           Si.append(simps(fi[i]**2.,Q[i])/L[i])
       #2nd-order variance
       Sij=[]
       Sij.append(doubleInteg(fij[0]**2.,Q[0],Q[1])/(L[0]*L[1]))
       Sij.append(doubleInteg(fij[1]**2.,Q[0],Q[2])/(L[0]*L[2]))
       Sij.append(doubleInteg(fij[2]**2.,Q[1],Q[2])/(L[1]*L[2]))
    else:
       print('ERROR in sobol_unif(): p>3 is currently unavailable!')
       
    #Sum of variances
    D=sum(Si)+sum(Sij)  
    #Main Sobol indices
    Si=Si/D
    Sij=Sij/D
    return Si,Sij

#/////////////////////////

    






##############
# Tests
##############
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
    Si,Sij=sobol_unif(q,fEx)

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
    Si_pce,Sij_pce=sobol_unif(qpceTest,fPCETest)

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
    Si,Sij=sobol_unif(q,fEx)

    print('sobol_3par_unif_test(): Sobol Sensitivity Indices for fEx3D("Ishigami")')
    print(' > Indices by UQit : S1=%g, S2=%g, S3=%g' %(Si[0],Si[1],Si[2]))
    print(' >                        S12=%g, S13=%g, S23=%g' %(Sij[0],Sij[1],Sij[2]))

    #(4) Exact Sobol indices (analytical expressions)
    fEx_.sobol(qBound)
    Si_ex=fEx_.Si
    Sij_ex=fEx_.Sij
    print(' > Analytical Reference: S1=%g, S2=%g, S3=%g' %(Si_ex[0],Si_ex[1],Si_ex[2]))
    print(' >                        S12=%g, S13=%g, S23=%g' %(Sij_ex[0],Sij_ex[1],Sij_ex[2]))
#    print(' > Analytical Expression: D1=%g, D2=%g, D12=%g' %(Si_ex[0],Si_ex[1],Sij_ex[0]))
