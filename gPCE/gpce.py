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

sys.path.append('../analyticFuncs/')
sys.path.append('../plot/')
import analyticTestFuncs
import plot2d

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
       Returns n Gauss-Legendre points over [-1,+1] along with associated weights for Gauss-lgenedre integration
       taken from: 
           https://pomax.github.io/bezierinfo/legendre-gauss.html
    """
    xGauss=np.zeros(n)
    w=np.zeros(n)

    if (n==1):
       xGauss[0]=0;
       w[0]=2.0;
    elif (n==2):
       xGauss[0]=-1.0/mt.sqrt(3.0);
       xGauss[1]=1.0/mt.sqrt(3.0);
       w[0]=1.0
       w[1]=1.0
    elif (n==3):
       xGauss[0]=-mt.sqrt(3.0/5.0);
       xGauss[1]=0.0;
       xGauss[2]=mt.sqrt(3.0/5.0);
       w[0]=5./9.
       w[1]=8./9.
       w[2]=5./9.
    elif (n==4):
       temp1=mt.sqrt(15.0-2.0*mt.sqrt(30.0))/mt.sqrt(35.0);
       temp2=mt.sqrt(15.0+2.0*mt.sqrt(30.0))/mt.sqrt(35.0);
       xGauss[0]=-temp2;
       xGauss[1]=-temp1;
       xGauss[2]= temp1;
       xGauss[3]= temp2;
       w[0]=49./(6.*(18.+mt.sqrt(30.)));
       w[1]=49./(6.*(18.-mt.sqrt(30.)));
       w[2]=49./(6.*(18.-mt.sqrt(30.)));
       w[3]=49./(6.*(18.+mt.sqrt(30.)));
    elif (n==5):
       temp1=mt.sqrt(5.0-2.0*mt.sqrt(10.0/7.0))/3.0;
       temp2=mt.sqrt(5.0+2.0*mt.sqrt(10.0/7.0))/3.0;
       xGauss[0]=-temp2;
       xGauss[1]=-temp1;
       xGauss[2]=0.0;
       xGauss[3]= temp1;
       xGauss[4]= temp2;
       w[0]=(322.-13.*mt.sqrt(70.))/900.;
       w[1]=(322.+13.*mt.sqrt(70.))/900.;
       w[2]=(128./225.);
       w[3]=(322.+13.*mt.sqrt(70.))/900.;
       w[4]=(322.-13.*mt.sqrt(70.))/900.;
    elif (n==6):
       xGauss[0]=-0.9324695142031521;
       xGauss[5]= 0.9324695142031521;
       xGauss[1]=-0.2386191860831969;
       xGauss[4]= 0.2386191860831969;
       xGauss[2]=-0.6612093864662645;
       xGauss[3]= 0.6612093864662645;
       w[0]= 0.1713244923791704;
       w[5]= w[0];
       w[1]=0.4679139345726910;
       w[4]=w[1];
       w[2]=0.3607615730481386;
       w[3]=w[2];
    elif (n==7):
       xGauss[0]=-0.9491079123427585;
       xGauss[6]= 0.9491079123427585;
       xGauss[1]=-0.7415311855993945;
       xGauss[5]= 0.7415311855993945;
       xGauss[2]= 0.4058451513773972;
       xGauss[4]=-0.4058451513773972;
       xGauss[3]=0.0;
       w[0]=0.1294849661688697;
       w[6]=0.1294849661688697;
       w[1]=0.2797053914892766;
       w[5]=w[1];
       w[2]=0.3818300505051189;
       w[4]=0.3818300505051189;
       w[3]=0.4179591836734694;
    elif (n==8):
       xGauss[0]=-0.960289856497536
       xGauss[1]=-0.796666477413626
       xGauss[2]=-0.525532409916329
       xGauss[3]=-0.183434642495649
       xGauss[4]=0.183434642495649
       xGauss[5]=0.525532409916329
       xGauss[6]=0.796666477413626
       xGauss[7]=0.960289856497536
       w[0]=0.101228536290376
       w[1]=0.222381034453374
       w[2]=0.313706645877887
       w[3]=0.362683783378362
       w[4]=0.362683783378362
       w[5]=0.313706645877887
       w[6]=0.222381034453374
       w[7]=0.101228536290376
    elif (n==9):
       xGauss[0]=-0.968160239507626
       xGauss[1]=-0.836031107326635
       xGauss[2]=-0.613371432700590
       xGauss[3]=-0.324253423403808
       xGauss[4]=0.000000000000000
       xGauss[5]=0.324253423403808
       xGauss[6]=0.613371432700590
       xGauss[7]=0.836031107326635
       xGauss[8]=0.968160239507626
       w[0]=0.081274388361574
       w[1]=0.180648160694857
       w[2]=0.260610696402935
       w[3]=0.312347077040002
       w[4]=0.330239355001259
       w[5]=0.312347077040002
       w[6]=0.260610696402935
       w[7]=0.180648160694857
       w[8]=0.081274388361574
    else:
       print('ERROR in GaussLeg_ptswts(): Invalid n.')

    return xGauss,w

#//////////////////////////////
def legendrePoly(m,x):
    """
       Standard Legendre polynomial of order m evaluated at x\in[-1,1] at x, where x is a d-dim in general
    """
    if (m==0):
       P=np.ones(x.size);
    elif (m==1):
       P=x;
    elif (m==2):    
       P=0.5*(3.*x**2.-1);
    elif (m==3):    
       P=0.5*(5.*x**3.-3.*x);    
    elif (m==4):    
       P=0.125*(35.*x**4.-30.*x**2.+3);    
    elif (m==5):    
       P=0.125*(63.*x**5.-70.*x**3.+15.*x);        
    elif (m==6):         
       P=0.5*0.125*(231.*x**6.-315.*x**4.+105.*x**2.-5);
    elif (m==7):         
       P=0.5*0.125*(429.*x**7.-693.*x**5.+315.*x**3.-35.*x);
    elif (m==8):         
       P=(1./128.)*(6435.*x**8.-12012*x**6.+6930.*x**4.-1260.*x**2.+35.);
    elif (m==9):         
       P=(1./128.)*(12155.*x**9.-25740*x**7.+18018.*x**5.-4620.*x**3.+315.*x);
    else:
       print('ERROR: number of samples is invalid legendrePoly()!')
    return P

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
        fVar+=fCoef[k]*fCoef[k]*sum2[k]
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
        fVar+=fCoef[k]*fCoef[k]*sum2[k]
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
        fVar+=fCoef[k]*fCoef[k]*sum2[k]
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
    qBound=[-.5,2.0]   #parameter bounds
    n=7   #number of Gauss samples
    nTest=100   #number of test sample sin the parameter space
    #--------------------------------------
    [xi,w]=GaussLeg_ptswts(n)   #Gauss sample pts in [-1,1]
    q=mapFromUnit(xi,qBound)    #map Gauss points to param space
    f=analyticTestFuncs.fEx1D(q)  #function value at the parameter samples (Gauss quads)
    fCoef,fMean,fVar=pce_LegUnif_1d_cnstrct(f)  #find PCE coefficients
    print('Mean of f(q) estimated by PCE = %g' %fMean)
    print('Var of f(q) estimated by PCE = %g' %fVar)
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
    nQ1=7      #number of collocation smaples of param1
    nQ2=7      #number of collocation smaples of param2
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
    q1Bound=[-1.0,2.0]   #range of param1
    q2Bound=[-3.0,3.0]   #range of param2
    q3Bound=[-1.0,1.0]   #range of param3
    nQ1=5      #number of collocation smaples of param1
    nQ2=5      #number of collocation smaples of param2
    nQ3=5      #number of collocation smaples of param3
    funOpt={'a':7.0,'b':0.1}   #parameters in Ishigami function
#    nTest1=100;   #number of test points in param1 space
#    nTest2=101;   #number of test points in param2 space
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
    m,v=analyticTestFuncs.ishigami_moments(q1Bound,q2Bound,q3Bound,funOpt)
    print(m,fMean,v,fVar)

