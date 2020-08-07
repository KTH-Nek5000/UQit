#################################################################
#            Compute Sobol Sensitivity Indices
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
    f1_=simps(f  ,Q[1],axis=1)/L[1]   #f1_=\int f  dq2  
    f1 =simps(f1_,Q[2],axis=1)/L[2]   #f1 =\int f1_dq3
    f2_=simps(f  ,Q[0],axis=0)/L[0]   #f2_=\int f  dq1
    f2 =simps(f2_,Q[2],axis=1)/L[2]   #f2 =\int f2_dq3
    f3_=simps(f  ,Q[0],axis=0)/L[0]   #f3_=\int f  dq1
    f3 =simps(f3_,Q[1],axis=0)/L[1]   #f3 =\int f3_dq2
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
    print('... Test: sobol_2par_unif_test(): Sobol Sensitivity Indices for fEx2D("type3")')
    def D1_Ex(a,b,q1):
        """
          Analytical function for variance D1 for for analyticalTestFuncs.fEx2D('type3')
          a,b: terms in derived expression f1(q1)=q1^2+a*q1+b
          q1: sample for q1 at which the expression is evaluated
        """
        return(0.2*q1**5.+0.5*a*q1**4.+(a**2.+2.*b)/3.*q1**3.+a*b*q1**2.+b**2.*q1)
        
    def D2_Ex(a,b,q2):
        """
          Analytical function for variance D2 for for analyticalTestFuncs.fEx2D('type3')
          a,b: terms in derived expression f2(q2)=a*q2+b
          q2: sample for q2 at which the expression is evaluated
        """
        return((a**2./3.)*q2**3.+a*b*q2**2.+b**2.*q2) 

    def D12_Ex(a,b,c,q1Bound,q2Bound):
        """
          Analytical function for variance D12 for for analyticalTestFuncs.fEx2D('type3')
          a,b: terms in derived expression f12(q1,q2)=q1*q2+a*q1+b*q2+c
          q1Bound, q2Bound: bounds of the samples at which the expression is evaluated
        """
        q1_1=q1Bound[1]-q1Bound[0]
        q1_2=q1Bound[1]**2.-q1Bound[0]**2.
        q1_3=q1Bound[1]**3.-q1Bound[0]**3.
        q2_1=q2Bound[1]-q2Bound[0]
        q2_2=q2Bound[1]**2.-q2Bound[0]**2.
        q2_3=q2Bound[1]**3.-q2Bound[0]**3.
        return(q1_3*q2_3/9.+a/3.*q1_3*q2_2+b*q1_2*q2_3/3.+0.5*(c+a*b)*q1_2*q2_2+a**2.*q1_3*q2_1/3.+b**2.*q1_1*q2_3/3.+a*c*q1_2*q2_1+b*c*q1_1*q2_2+c**2.*q1_1*q2_1)

    def analyticalSobol_2par(qBound):
        """
        Analytical Sobol indices for analyticalTestFuncs.fEx2D('type3')
        """      
        #Variances in Sobol decomposition
        a1=qBound[0][0]
        b1=qBound[0][1]
        a2=qBound[1][0]
        b2=qBound[1][1]
        f0=(a1**2.+b1**2.+a1*b1)/3.+0.25*(a1+b1)*(a2+b2)
        a=(a2+b2)/2.0
        b=-f0
        D1_ex=(D1_Ex(a,b,b1)-D1_Ex(a,b,a1))/(b1-a1)
        a=(a1+b1)/2.0
        b=(a1**2.+a1*b1+b1**2.)/3.-f0
        D2_ex=(D2_Ex(a,b,b2)-D2_Ex(a,b,a2))/(b2-a2)
        a=-(a2+b2)/2.
        b=-(a1+b1)/2.
        c=f0-(a1**2.+a1*b1+b1**2.)/3.
        D12_ex=(D12_Ex(a,b,c,qBound[0],qBound[1]))/((b1-a1)*(b2-a2))
        #Sensitivity indices
        D_ex=D1_ex+D2_ex+D12_ex
        Si_ex=[D1_ex/D_ex,D2_ex/D_ex]
        Sij_ex=[D12_ex/D_ex]
        return Si_ex,Sij_ex
       
    #--------------------------
    #------- SETTINGS 
    n=[100, 90]       #number of samples for q1 and q2
    qBound=[[-3,1],   #admissible range of parameters
            [-1,2]]
    #--------------------------
    p=len(n)
    #(1) Samples from parameters space
    q=[]
    for i in range(p):
        q.append(np.linspace(qBound[i][0],qBound[i][1],n[i]))

    #(2) Compute function value at the parameter samples
    fEx_=analyticTestFuncs.fEx2D(q[0],q[1],'type3','tensorProd')    
    fEx=np.reshape(fEx_,(n[0],n[1]),'F')  
    
    #(3) Compute Sobol indices direct numerical integration
    Si,Sij=sobol_unif([q[0],q[1]],fEx)

    #(4) Construct a gPCE and then use the predictions of the gPCE in numerical integration for computing Sobol indices.
    #generate observations at Gauss-Legendre points
    nQpce=[5,6]
    [xi1,w1]=pce.GaussLeg_ptswts(nQpce[0])
    [xi2,w2]=pce.GaussLeg_ptswts(nQpce[1])
    q1pce=pce.mapFromUnit(xi1,qBound[0])   
    q2pce=pce.mapFromUnit(xi2,qBound[1])    
    fVal_pceCnstrct=analyticTestFuncs.fEx2D(q1pce,q2pce,'type3','tensorProd') 
    #construct the gPCE
    xiGrid=reshaper.vecs2grid(xi1,xi2)
    pceDict={'sampleType':'GQ','truncMethod':'TP','pceSolveMethod':'Projection'}
    pceDict=pce.pceDict_corrector(pceDict)
    fCoefs,kSet,fMean,fVar=pce.pce_LegUnif_2d_cnstrct(fVal_pceCnstrct,nQpce,xiGrid,pceDict)
    #use gPCE to predict at test samples from parameter space
    q1pceTest =np.linspace(qBound[0][0],qBound[0][1],n[0])  
    xi1Test   =pce.mapToUnit(q1pceTest,qBound[0])
    q2pceTest =np.linspace(qBound[1][0],qBound[1][1],n[1])  
    xi2Test   =pce.mapToUnit(q2pceTest,qBound[1])
    fPCETest  =pce.pce_LegUnif_2d_eval(fCoefs,kSet,xi1Test,xi2Test)
    #compute Sobol indices 
    Si_pce,Sij_pce=sobol_unif([q1pceTest,q2pceTest],fPCETest)

    #(5) Exact Sobol indices (analytical expressions)
    Si_ex,Sij_ex=analyticalSobol_2par(qBound)
    
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
    fEx=np.reshape(fEx_,(n[0],n[1],n[2]),'F')  
    
    #(3) Compute Sobol indices (method of choice in this library)
    Si,Sij=sobol_unif([q[0],q[1],q[2]],fEx)

    #(4) Exact Sobol indices (analytical expressions)
##    Si_ex,Sij_ex=analyticalSobol_2par(qBound)
    
    #(5) Write results on screen
    print('sobol_3par_unif_test(): Sobol Sensitivity Indices for fEx3D("Ishigami")')
    print(' > Indices by UQit : S1=%g, S2=%g, S3=%g' %(Si[0],Si[1],Si[2]))
    print(' >                        S12=%g, S13=%g, S23=%g' %(Sij[0],Sij[1],Sij[2]))

    D1=b*pi**4./5.+b**2.*pi**8./50. + 0.5
    D2=a**2./8.0
    D3=0.0
    D13=b**2.*pi**8./50.0+7.*b**2.*pi**8./450
    D12=0.0
    D23=0.0
    D123=0.0
  
    D=D1+D2+D3+D12+D13+D23+D123
    Si_ex=[D1/D,D2/D,D3/D]
    Sij_ex=[D12/D,D13/D,D23/D]
    print(' > Analytical Reference: S1=%g, S2=%g, S3=%g' %(Si_ex[0],Si_ex[1],Si_ex[2]))
    print(' >                        S12=%g, S13=%g, S23=%g' %(Sij_ex[0],Sij_ex[1],Sij_ex[2]))
#    print(' > Analytical Expression: D1=%g, D2=%g, D12=%g' %(Si_ex[0],Si_ex[1],Sij_ex[0]))
