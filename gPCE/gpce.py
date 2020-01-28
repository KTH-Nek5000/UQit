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
sys.path.append('../general/')
import analyticTestFuncs
import plot2d
import writeUQ
import reshaper
import linAlg

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
def pceDict_corrector(pceDict):
    """
       Correct pceDict for PCE over multi-dimensional parameter space, so that options become consistent. 
         * For 'GQ' samples+'TP' truncation method: either 'Projection' or 'Regression' can be used
         * For all combination of sample points and truncation, 'Projection' can be used to compute PCE coefficients
    """
    if pceDict['truncMethod']=='TO':
       pceDict['pceSolveMethod']='Regression'
    if pceDict['truncMethod']=='TP' and pceDict['sampleType']!='GQ':
       pceDict['pceSolveMethod']='Regression'
    if pceDict['pceSolveMethod']=='Regression' and pceDict['truncMethod']!='TP':
       if 'LMax' not in pceDict:
          print("ERROR in pceDict: 'LMax' should be set when Total-Order method is used.")
    return pceDict

#////////////////////////////////////
def PCE_coef_conv_plot(fCoef,kSet,distType):
    """
       Plot convergence of PCE terms
       ||fk*Psi_k||/||f0*Psi_0|| is plotted versus |k|=sum(k_i)
       Inputs:
          fCoef: 1D array of length K containing a PCE coefficients
          kSet: Index set of PCE, kSet=[[k1,k2,...,kp],...]
          distType: A list containing distribution type of RVs, distType=[dist1,dist2,...,distp]      
    """
    K=len(fCoef)   #no of terms in PCE
    p=len(kSet[0]) #dimension of parameter-space
    #magnitude of indices
    kMag=[]
    for i in range(K):
        kMag.append(sum(kSet[i]))
    #compute norm of the PCE bases
    xi=np.linspace(-1,1,1000)
    termNorm=[]
    for ik in range(K):   #over PCE terms
        PsiNorm=1.0
        for ip in range(p):   #over parameter dimension 
            k_=kSet[ik][ip]
            if distType[ip]=='Unif':   
               psi_k_=legendrePoly(k_,xi)
               PsiNorm*=np.linalg.norm(psi_k_,2)
            else:
               print('NOW only uniform distribution is considered!')
        termNorm.append(abs(fCoef[ik])*PsiNorm)
    termNorm0=termNorm[0]
    plt.semilogy(kMag,termNorm/termNorm0,'ob')
    plt.xlabel(r'$|\mathbf{k}|$',fontsize=18)
    plt.ylabel(r'$||\hat{f}_\mathbf{k}\Psi_{\mathbf{k}(\mathbf{\xi})}||_2/||\hat{f}_0||_2$',fontsize=18)
    plt.grid()
    #plt.ylim([1e-40,1.1])


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

#///////////////////////////////////////////////////
def pce_LegUnif_2d_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 2D parameter space. 
       Uniform Uncertain Parameter
       => Legendre Polynomials
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
          Input:
             fVal: 1d numpy array containing response value at the samples
             nQList: A list containing number of samples, nQList=[nQ1|nQ2]
                     For Tensor Product method: nQ1,nQ2 are number of samples in each direction
                     For Other methods: nQ1=nData (total no of samples), nQ2=1
             xiGrid: A 2D grid of sampled parameters mapped in [-1,1]^2 space
                     xiGrid=2d np array, [nData,2]
                     Always Required to be provided unless 'GQ' samples are used
             pceDict: A dictionary containing different options for PCE with these keys:   
                    'truncMethod': method of truncating PCE
                                 ='TP' (tensor product method)         
                                 ='TO' (total order method)
                    'pceSolveMethod': method of solving for PCE coeffcients
                                 ='Projection' (Requires samples to be Gauss-Quadratures)
                                 ='Regression' (can be uniquely-, over-, and under-determined. In the latter compressed sensing with L1/L2 regularization is needed.)                                                  NOTE: for 'GQ'+'TP' the pceSolveMethod is 'Projection'. For any other combination, we use 'Regression'
                    'sampleType': type of parameter samples at which observations are made
                                ='GQ' (Gauss Quadrature nodes)
                                =' '  (Any other nodal set)
          Output:
             fCoef: Coefcients in the PCE: length =K
             kSet:  Index set, list of pd lists
             fMean: PCE estimation for E[f(q)]
             fVar:  PCE estimation for V[f(q)]
    """
    truncMethod=pceDict['truncMethod']   #Truncation method for PCE
    sampleType=pceDict['sampleType']     #Types of parameter samples
    if sampleType=='GQ' and truncMethod=='TP':   #Gauss Quadrature samples with Tensor-Product Rules
       fCoef,kSet,fMean,fVar=pce_LegUnif_2d_GQTP_cnstrct(fVal,nQList,pceDict)
    else:                  #Other type of samples (structured/unstructured)
       fCoef,kSet,fMean,fVar=pce_LegUnif_2d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict)
    return fCoef,kSet,fMean,fVar

#//////////////////////////////////////////////////
def pce_LegUnif_2d_GQTP_cnstrct(fVal,nQList,pceDict):
    """ 
       Construct a PCE over a 2D parameter space 
       Uniform Uncertain Parameter=> Legendre Polynomials, 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       * Type of parameter samples: Gauss-Quadrature nodes
       * Method of truncating PCE: Tensor Product Method
         pceSolveMethod= 'Projection' or 'Regression 
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    print('... A gPCE for a 2D parameter space is constructed.')
    print('       Samples in each direction are Gauss Quadrature nodes (User should check this!).')
    print('       PCE is truncated using Tensor Product method.')
    print('       Coeffcients are computed by %s method.' %pceDict['pceSolveMethod'])
    #(1) Set variables
    nQ1=nQList[0]
    nQ2=nQList[1]
    [xi1,w1]=GaussLeg_ptswts(nQ1)
    [xi2,w2]=GaussLeg_ptswts(nQ2)
    K=nQ1*nQ2;  #upper bound of sum in PCE using Tensor Product truncation
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    if K!=nData:
       print('ERROR in pce_LegUnif_2d_TP_cnstrct(): ')
    #(2) Find the coefficients in the expansion
    #By default, Projection method is used (assuming samples are Gauss-Quadrature points)
    fCoef=np.zeros(K)
    sum2=[]
    kSet=[]    #index set
    for k2 in range(nQ2):  #k-th coeff in PCE - param2
        psi_k2=legendrePoly(k2,xi2)
        for k1 in range(nQ1):  #k-th coeff in PCE - param1
            psi_k1=legendrePoly(k1,xi1)
            sum1=0.0
            sum2_=0.0
            k=k2*nQ1+k1
            kSet.append([k1,k2])
            for j2 in range(nQ2):
                for j1 in range(nQ1):
                    j=j2*nQ1+j1
                    sum1 += fVal[j]*(psi_k1[j1]*psi_k2[j2]*w1[j1]*w2[j2])
                    sum2_+=         (psi_k1[j1]*psi_k2[j2])**2.*w1[j1]*w2[j2]
            fCoef[k]=sum1/sum2_
            sum2.append(sum2_)

    #(2b) Recompute fCoef via Regression, in case Regression is chosen
    if pceDict['pceSolveMethod']=='Regression':
       xiGrid=reshaper.vecs2grid(xi1,xi2)
       A=np.zeros((nData,K))
       k=-1
       for k2 in range(nQ2): 
           for k1 in range(nQ1): 
               k+=1
               for j in range(nData):
                   A[j,k]=legendrePoly(k1,xiGrid[j,0])*legendrePoly(k2,xiGrid[j,1])
       #Linear Regression to solve the linear set of equations
       fCoef=linAlg.myLinearRegress(A,fVal)   #This is a uniquely determined system
    
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**2.)*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,kSet,fMean,fVar

#//////////////////////////////////////////////////
def pce_LegUnif_2d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 2D parameter space 
       Uniform Uncertain Parameter=> Legendre Polynomials, 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Inputs:
           xiGrid: sampled parameters mapped to [-1,1]^2  
                  A Grid of nData x 2
                  NOTE: xiGrid is ALWAYS needed in this routine
           nQList: only needed if 'TP' is selected
       * Method of truncating PCE: Total Order or Tensor Product
       * Method of solving for PCE coefficients: ONLY Regression
       (Any combination but 'GQ'+'TP')
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    pceSolveMethod=pceDict['pceSolveMethod']
    truncMethod=pceDict['truncMethod']
    print('... A gPCE for a 2D parameter space is constructed.')
    print('      PCE is truncated using %s method.' %truncMethod)    
    print('      Coeffcients are computed by %s method,' %pceSolveMethod)
    if pceDict['truncMethod']=='TO':
       LMax=pceDict['LMax']   #max order of polynomial in each direction
       print('         with LMax=%d as the max polynomial order in each direction.' %LMax)
    if pceSolveMethod!='Regression':
       print('ERROR in pce_LegUnif_2d_TO_cnstrct(): only Regression method can be used for PCE with Total Order truncation method.')

    #(1) Set variables
    #Number of terms in PCE
    if truncMethod=='TO':
       K=int((LMax+2)*(LMax+1)/2)  
       N1=LMax+1  #loops' upper bound
       N2=LMax+1
    elif truncMethod=='TP':   #'TP' needs nQList 
       N1=nqList[0]
       N2=nqList[1]
       K=N1*N2           
       
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    #(2) Find the coefficients in the expansion:Only Regression method can be used. 
    #    Also we need to compute gamma_k (=sum2) required to estimate the variance by PCE.
    #    For this we create an auxiliary Gauss-Quadrature grid to compute intgerals
    A=np.zeros((nData,K))    #Matrix of known coeffcient for regression to compute PCE coeffcients
    kSet=[]
    sum2=[]
    k=-1
    xi1_aux,w1_aux=GaussLeg_ptswts(N1)  #auxiliary GQ rule for computing gamma_k
    xi2_aux,w2_aux=GaussLeg_ptswts(N2)
    for k2 in range(N2):    
        psi_aux_k2=legendrePoly(k2,xi2_aux)
        for k1 in range(N1):
            psi_aux_k1=legendrePoly(k1,xi1_aux)
            if (truncMethod=='TO' and (k1+k2)<=LMax) or (truncMethod=='TP'):
               #constructing Aij
               k+=1
               kSet.append([k1,k2])    #index set
               for j in range(nData):
                   A[j,k]=legendrePoly(k1,xiGrid[j,0])*legendrePoly(k2,xiGrid[j,1])
               #computing sum2=gamma_k
               sum2_=0.0
               k_aux=k2*(N1)+k1  
               for j2 in range(N2):
                   for j1 in range(N1):
                       sum2_+=(psi_aux_k1[j1]*psi_aux_k2[j2])**2.*w1_aux[j1]*w2_aux[j2]
               sum2.append(sum2_)

    #Find the PCE coeffs by Linear Regression 
    fCoef=linAlg.myLinearRegress(A,fVal)   #This can be over-, under-, or uniquely- determined systetm.
    
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**2.)*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,kSet,fMean,fVar

#//////////////////////////////
def pce_LegUnif_2d_eval(fk,kSet,xi1,xi2):
    """ 
       Evaluate a PCE over 2D param space at a set of test points xi1,xi2\in[-1,1] 
       i.e., Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
       Note: This routine works for any PCE no matter what truncation method has been used for its construction. Because, we use the index set kSet.
        Uniform Uncertain Parameter    => Legendre Polynomials
        Inputs:
           fk: 1D array of length K containing the coefficients of the PCE
           kSet: List of indices:[[k1,1,k2,1],[k1,2,k2,2],...,[k1,K,k2,K]] produced based on the tensor product or total order rules when constructing the PCE
           xi1,xi2: Test points in each direction of the 2D parameter space. Always a tensor product grid is constructed based on test points to evaluate the PCE.
        Outputs: 
           fpce: Response values predicted (inerpolated) by the PCE at the test points
    """
    xi1 = np.array(xi1, copy=False, ndmin=1)
    xi2 = np.array(xi2, copy=False, ndmin=1)
    n1=xi1.size
    n2=xi2.size
    fpce=np.zeros((n1,n2))
    K=len(kSet)   #number of terms in the PCE
    for i2 in range(n2):
        for i1 in range(n1):
            sum1=0.0;
            for k in range(K):
                k1=kSet[k][0]
                k2=kSet[k][1]
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
    nQ1=6      #number of collocation smaples of param1
    nQ2=7      #number of collocation smaples of param2
    nTest1=100;   #number of test points in param1 space
    nTest2=101;   #number of test points in param2 space
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType=''     #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'TO' can be selected
    pceSolveMethod='Projection' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=12   #max polynomial order in each parameter direction
    #--------------------

    #generate observations   
    [xi1,w1]=GaussLeg_ptswts(nQ1)   #Gauss sample pts in [-1,1]
    [xi2,w2]=GaussLeg_ptswts(nQ2)   #Gauss sample pts in [-1,1]
    q1=mapFromUnit(xi1,q1Bound)    #map Gauss points to param1 space
    q2=mapFromUnit(xi2,q2Bound)    #map Gauss points to param2 space
    fVal=analyticTestFuncs.fEx2D(q1,q2,'type1','tensorProd')  #function value at the parameter samples (Gauss quads)    
    #construct the gPCE
    pceDict={'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    pceDict=pceDict_corrector(pceDict)
    xiGrid=reshaper.vecs2grid(xi1,xi2)
    fCoefs,kSet,fMean,fVar=pce_LegUnif_2d_cnstrct(fVal,[nQ1,nQ2],xiGrid,pceDict)

    #plot convergence of PCE terms
    PCE_coef_conv_plot(fCoefs,kSet,['Unif','Unif'])

    #make predictions at test points in the parameter space
    q1Test =np.linspace(q1Bound[0],q1Bound[1],nTest1)  #test points in param1 space
    xi1Test=mapToUnit(q1Test,q1Bound)
    q2Test =np.linspace(q2Bound[0],q2Bound[1],nTest2)  #test points in param2 space
    xi2Test=mapToUnit(q2Test,q2Bound)
    fTest=analyticTestFuncs.fEx2D(q1Test,q2Test,'type1','tensorProd')   #response value at the test points
    fPCE=pce_LegUnif_2d_eval(fCoefs,kSet,xi1Test,xi2Test)  #Prediction at test points by PCE
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
