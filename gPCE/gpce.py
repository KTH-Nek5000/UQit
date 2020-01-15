#############################################
# generalized Polynomial Chaos Expansion
#############################################
# There are tests as external functions
# Note1: in multi-dim parameter space:
#     - in the current version we only consider tensor product
#     - always the last parameter is the outer loop when reading/writing
#     - always the response is a vector with size nSample1*nSample2*...
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
#from numpy.polynomial import legendre


sys.path.append('../analyticFuncs/')
sys.path.append('../plot/')
sys.path.append('../writeUQ/')
import analyticTestFuncs
import plot2d
import writeUQ

#////////////////////////////
def mapToUnit(x,xBound):
    """
    Linearly map x\in[xBound] to xi\in[-1,1]
    x can be either scalar or a vector
    """
    x = np.array(x, copy=False, ndmin=1)
    xi=(2.*(x[:]-xBound[0])/(xBound[1]-xBound[0])-1.)
    return xi

#////////////////////////////
def mapFromUnit(xi,xBound):
    """
    Linearly map xi\in[-1,1] to x\in[xBound]
    x can be either scalar or a vector
    """
    xi = np.array(xi, copy=False, ndmin=1)
    x=(0.5*(xi[:]+1.0)*(xBound[1]-xBound[0])+xBound[0])
    return x

#////////////////////////////
def GaussLeg_ptswts(n):
    """
       Returns n Gauss-Legendre Quadrature points over [-1,1] and associated Weights
    """
    x=np.polynomial.legendre.leggauss(n)
    quads=x[0]
    weights=x[1]
    return quads,weights


#//////////////////////////////
def legendrePoly(n,xi):
    """
        Evaluate Legendre polynomial of order n at xi\in[-1,1]
    """
    return np.polynomial.legendre.legval(xi,[0]*n+[1])

#//////////////////////////////
def pce_LegUnif_1d_cnstrct(fVal):
    """ 
    Construct a PCE over a 1D parameter space. 
    Uniform Uncertain Parameter
    => Legendre Polynomials
    Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
    """
    nQ=len(fVal) #number of quadratures (collocation samples)
    [xi,w]=GaussLeg_ptswts(nQ)
    K=nQ;  #upper bound of sum in PCE
    #find the coefficients in the expansion
    fCoef=np.zeros(nQ)
    sum2=[]
    for k in range(K):  #k-th coeff in PCE
        psi_k=legendrePoly(k,xi)
        sum1=0.0
        sum2_=0.0
        for j in range(K):
            sum1+=fVal[j]*psi_k[j]*w[j]
            sum2_+=psi_k[j]*psi_k[j]*w[j]
        fCoef[k]=sum1/sum2_
        sum2.append(sum2_)
    #find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=0.5*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of U on [-1,1]
    return fCoef,fMean,fVar

#//////////////////////////////
def pce_LegUnif_1d_eval(fk,xi):
    """ 
    Evaluate a 1D PCE at a set of test points xi\in[-1,1]
    Uniform Uncertain Parameter
    => Legendre Polynomials
    Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
    """
    K=len(fk) 
    xi = np.array(xi, copy=False, ndmin=1)
    fpce=[]
    for i in range(xi.size):
        sum1=0.0;
        for k in range(K):
            sum1+=fk[k]*legendrePoly(k,xi[i])
        fpce.append(sum1)
    return fpce

#//////////////////////////////
def pce_LegUnif_2d_cnstrct(fVal,nQ1,nQ2):
    """ 
    Construct a PCE over a 2D parameter space. 
    Uniform Uncertain Parameter
    => Legendre Polynomials
    Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
        nQ1,nQ2: number of parameter samples for 1st and 2nd parameters
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    [xi1,w1]=GaussLeg_ptswts(nQ1)
    [xi2,w2]=GaussLeg_ptswts(nQ2)
    K=nQ1*nQ2;  #upper bound of sum in PCE
                #NOTE: assuming Tensor Product
    #find the coefficients in the expansion
    fCoef=np.zeros(K)
    sum2=[]
    for k2 in range(nQ2):  #k-th coeff in PCE - param2
        psi_k2=legendrePoly(k2,xi2)
        for k1 in range(nQ1):  #k-th coeff in PCE - param1
             psi_k1=legendrePoly(k1,xi1)
             sum1=0.0
             sum2_=0.0
             k=k2*nQ1+k1
             for j2 in range(nQ2):
                 for j1 in range(nQ1):
                     j=j2*nQ1+j1
                     sum1 += fVal[j]*(psi_k1[j1]*psi_k2[j2]*w1[j1]*w2[j2])
                     sum2_+=         (psi_k1[j1]*psi_k2[j2])**2.*w1[j1]*w2[j2]
             fCoef[k]=sum1/sum2_
             sum2.append(sum2_)
    #find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**2.)*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,fMean,fVar

#//////////////////////////////
def pce_LegUnif_2d_eval(fk,nQ1,nQ2,xi1,xi2):
    """ 
    Evaluate a 2D PCE at a set of test points xi1,xi2\in[-1,1] which are assumed to make a tensor-product grid
    Uniform Uncertain Parameter
    => Legendre Polynomials
    Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
    #NOTE: assumes Tensor product
    """
    xi1 = np.array(xi1, copy=False, ndmin=1)
    xi2 = np.array(xi2, copy=False, ndmin=1)
    n1=xi1.size
    n2=xi2.size
    fpce=np.zeros((n1,n2))
    for i2 in range(n2):
        for i1 in range(n1):
            sum1=0.0;
            for k2 in range(nQ2):
                for k1 in range(nQ1):
                    k=k2*nQ1+k1
                    sum1+=fk[k]*legendrePoly(k1,xi1[i1])*legendrePoly(k2,xi2[i2])
            fpce[i1,i2]=sum1
    return fpce


#//////////////////////////////
def pce_LegUnif_3d_cnstrct(fVal,nQ1,nQ2,nQ3):
    """ 
    Construct a PCE over a 3D parameter space. 
    Uniform Uncertain Parameter
    => Legendre Polynomials
    Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
        nQ1,nQ2, nQ3: number of parameter samples for 1st, 2nd, and 3rd parameters
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    [xi1,w1]=GaussLeg_ptswts(nQ1)
    [xi2,w2]=GaussLeg_ptswts(nQ2)
    [xi3,w3]=GaussLeg_ptswts(nQ3)
    K=nQ1*nQ2*nQ3;  #upper bound of sum in PCE
                    #NOTE: assuming Tensor Product
    #find the coefficients in the expansion
    fCoef=np.zeros(K)
    sum2=[]
    for k3 in range(nQ3):      #k-th coeff in PCE - param3
        psi_k3=legendrePoly(k3,xi3)
        for k2 in range(nQ2):  #k-th coeff in PCE - param2
            psi_k2=legendrePoly(k2,xi2)
            for k1 in range(nQ1):  #k-th coeff in PCE - param1
                psi_k1=legendrePoly(k1,xi1)
                sum1=0.0
                sum2_=0.0
                k=(k3*nQ2*nQ1)+(k2*nQ1)+k1
                for j3 in range(nQ3):
                    for j2 in range(nQ2):
                        for j1 in range(nQ1):
                            j=(j3*nQ2*nQ1)+(j2*nQ1)+j1
                            sum1 += fVal[j]*(psi_k1[j1]*psi_k2[j2]*psi_k3[j3]*w1[j1]*w2[j2]*w3[j3])
                            sum2_+=         (psi_k1[j1]*psi_k2[j2]*psi_k3[j3])**2.*w1[j1]*w2[j2]*w3[j3]
                fCoef[k]=sum1/sum2_
                sum2.append(sum2_)
    #find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**3.)*fCoef[k]*fCoef[k]*sum2[k]    #0.5:PDF of U on [-1,1]
    return fCoef,fMean,fVar

#///////////////////////////////////////////
def pce_LegUnif_3d_eval(fk,nQ1,nQ2,nQ3,xi1,xi2,xi3):
    """ 
       Evaluate a 3D PCE at a set of test points xi1,xi2,xi3\in[-1,1] which are assumed to make a tensor-product grid
         Uniform Uncertain Parameter
         => Legendre Polynomials
        Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
       #NOTE: assumes Tensor product
    """
    xi1 = np.array(xi1, copy=False, ndmin=1)
    xi2 = np.array(xi2, copy=False, ndmin=1)
    xi3 = np.array(xi3, copy=False, ndmin=1)
    n1=xi1.size
    n2=xi2.size
    n3=xi3.size
    fpce=np.zeros((n1,n2,n3))
    for i3 in range(n3):
        for i2 in range(n2):
            for i1 in range(n1):
                sum1=0.0;
                for k3 in range(nQ3):
                    for k2 in range(nQ2):
                        for k1 in range(nQ1):
                            k=(k3*nQ2*nQ1)+(k2*nQ1)+k1
                            sum1+=fk[k]*legendrePoly(k1,xi1[i1])*legendrePoly(k2,xi2[i2])*legendrePoly(k3,xi3[i3])
                fpce[i1,i2,i3]=sum1
    return fpce
    

################################
# external functions
################################ 
#--------------------------------------------------
#To run a test:
#python -c 'import gpce as X;X.gpce_test_1d()'
#--------------------------------------------------

#A set of tests for gPCE methods

#/////////////////
def gpce_test_1d():
    """
    Test PCE for 1 uncertain parameter uniformly distributed over [a1,b1] using Legendre polynomial bases
    """
    print('------ gPCE TEST 1 ------')
    #--- settings -------------------------
    qBound=[-2,4.0]   #parameter bounds
    n=20   #number of Gauss samples
    nTest=100   #number of test sample sin the parameter space
    #--------------------------------------
    #compute exact moments
    fMean_ex,fVar_ex=analyticTestFuncs.fEx1D_moments(qBound)

    #construct the PCE
    [xi,w]=GaussLeg_ptswts(n)   #Gauss sample pts in [-1,1]
    q=mapFromUnit(xi,qBound)    #map Gauss points to param space
    f=analyticTestFuncs.fEx1D(q)  #function value at the parameter samples (Gauss quads)
    fCoef,fMean,fVar=pce_LegUnif_1d_cnstrct(f)  #find PCE coefficients
    print('-------------- Exact -------- PCE --------- Error % ') 
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_ex,fMean,(fMean-fMean_ex)/fMean_ex*100.))
    print('Var of f(q) = %g\t%g\t%g' %(fVar_ex,fVar,(fVar-fVar_ex)/fVar_ex*100.))
    #plot
    qTest=np.linspace(qBound[0],qBound[1],nTest)  #test points in param space
    fTest=analyticTestFuncs.fEx1D(qTest)   #exact response at test points
    xiTest=mapToUnit(qTest,qBound)
    fPCE=pce_LegUnif_1d_eval(fCoef,xiTest)  #Prediction by PCE
    plt.figure(figsize=(20,8));
    ax=plt.gca();
    plt.plot(qTest,fTest,'-k',lw=2,label=r'Exact $f(q)$')
    plt.plot(q,f,'ob',label='Gauss-Legendre Samples')
    plt.plot(qTest,fPCE,'-r',lw=2,label='PCE')
    plt.plot(qTest,fMean*np.ones(len(qTest)),'-b',label=r'$\mathbb{E}(f(q))$') 
    ax.fill_between(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),color='powderblue',alpha=0.4)
    plt.plot(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b',label=r'$\mathbb{E}(f(q))\pm 95\%CI$')
    plt.plot(qTest,fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b')
    plt.title('Example of 1D PCE for uniform random variable')
    plt.xlabel(r'$q$',fontsize=26)
    plt.ylabel(r'$f(q)$',fontsize=26)
    plt.grid()
    plt.legend()
    plt.show()


#/////////////////
def gpce_test_2d():
    """
    Test PCE for 2 uncertain parameters uniformly distributed over [a1,b1]x[a2,b2] using Legendre polynomial bases
    """
    print('------ gPCE TEST 2 ------')
    #settings------------
    q1Bound=[-1.0,2.0]   #range of param1
    q2Bound=[-3.0,3.0]     #range of param2
    nQ1=11      #number of collocation smaples of param1
    nQ2=6      #number of collocation smaples of param2
    nTest1=100;   #number of test points in param1 space
    nTest2=101;   #number of test points in param2 space
    #--------------------
    #generate observations   
    [xi1,w1]=GaussLeg_ptswts(nQ1)   #Gauss sample pts in [-1,1]
    [xi2,w2]=GaussLeg_ptswts(nQ2)   #Gauss sample pts in [-1,1]
    q1=mapFromUnit(xi1,q1Bound)    #map Gauss points to param1 space
    q2=mapFromUnit(xi2,q2Bound)    #map Gauss points to param2 space
    fVal=analyticTestFuncs.fEx2D(q1,q2,'type1','tensorProd')  #function value at the parameter samples (Gauss quads)    
    #construct the gPCE
    fCoefs,fMean,fVar=pce_LegUnif_2d_cnstrct(fVal,nQ1,nQ2)
    #make predictions at test points in the parameter space
    q1Test =np.linspace(q1Bound[0],q1Bound[1],nTest1)  #test points in param1 space
    xi1Test=mapToUnit(q1Test,q1Bound)
    q2Test =np.linspace(q2Bound[0],q2Bound[1],nTest2)  #test points in param2 space
    xi2Test=mapToUnit(q2Test,q2Bound)
    fTest=analyticTestFuncs.fEx2D(q1Test,q2Test,'type1','tensorProd')   #response value at the test points
    fPCE=pce_LegUnif_2d_eval(fCoefs,nQ1,nQ2,xi1Test,xi2Test)  #Prediction at test points by PCE
    #create 2D grid and response surface over it
    x1TestGrid,x2TestGrid,fTestGrid=plot2d.plot2D_gridVals(q1Test,q2Test,fTest)
    fErrorGrid=np.zeros((nTest1,nTest2))
    for j in range(nTest2):
        for i in range(nTest1):
            k=i+j*nTest1
            #compute error between exact and surrogate response
            tmp=fTestGrid[i,j]
            if abs(tmp)<1.e-1:
               tmp=1e-1
            fErrorGrid[i,j]=((abs(fTestGrid[i,j]-fPCE[i,j]))/tmp*100.)

    #2d grid from the sampled parameters
    q1Grid,q2Grid=plot2d.plot2D_grid(q1,q2)

    #plot 2d contours
    plt.figure(figsize=(21,8));
    plt.subplot(1,3,1)
    ax=plt.gca()
    CS1 = plt.contour(x1TestGrid,x2TestGrid,fTestGrid,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact Response')

    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(x1TestGrid,x2TestGrid,fPCE,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Surrogate Response')

    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(x1TestGrid,x2TestGrid,fErrorGrid,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.0f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.title('|Exact-Surrogate|/Exact %')
    plt.show()
     
#/////////////////
def gpce_test_3d():
    """
    Test PCE for 3 uncertain parameters uniformly distributed over [a1,b1]x[a2,b2]x[a3,b3] using Legendre polynomial bases
    """
    print('------ gPCE TEST 3 ------')
    #settings------------
    q1Bound=[-0.75,1.5]   #range of param1
    q2Bound=[-0.5,2.5]   #range of param2
    q3Bound=[ 1.0,3.0]   #range of param3
    nQ1=4      #number of collocation smaples of param1
    nQ2=4      #number of collocation smaples of param2
    nQ3=3      #number of collocation smaples of param3
    funOpt={'a':7,'b':0.1}   #parameters in Ishigami function
    #--------------------
    #generate observations   
    [xi1,w1]=GaussLeg_ptswts(nQ1)   #Gauss sample pts in [-1,1]
    [xi2,w2]=GaussLeg_ptswts(nQ2)   #Gauss sample pts in [-1,1]
    [xi3,w3]=GaussLeg_ptswts(nQ3)   #Gauss sample pts in [-1,1]
    q1=mapFromUnit(xi1,q1Bound)    #map Gauss points to param1 space
    q2=mapFromUnit(xi2,q2Bound)    #map Gauss points to param2 space
    q3=mapFromUnit(xi3,q3Bound)    #map Gauss points to param3 space
    fVal=analyticTestFuncs.fEx3D(q1,q2,q3,'Ishigami','tensorProd',funOpt)  #function value at the parameter samples (Gauss quads)    
    #construct the gPCE and compute the moments
    fCoefs,fMean,fVar=pce_LegUnif_3d_cnstrct(fVal,nQ1,nQ2,nQ3)
    #exact moments of Ishigami function
    m,v=analyticTestFuncs.ishigami_exactMoments(q1Bound,q2Bound,q3Bound,funOpt)
    #print the results
    print(writeUQ.printRepeated('-',50))
    print('\t\t Exact \t\t PCE')
    print('E[f]:  ',m,fMean)
    print('V[f]:  ',v,fVar)
