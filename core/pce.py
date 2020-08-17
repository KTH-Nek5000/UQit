#############################################
# generalized Polynomial Chaos Expansion
#############################################
# There are tests as external functions
# Note1: in multi-dim parameter space:
#     - always the last parameter is the outer loop when reading/writing
#     - always the response is a vector with size nSample1*nSample2*...
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
import os
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
UQit=os.getenv("UQit")
sys.path.append(UQit)
import analyticTestFuncs
import plot2d
import writeUQ
import reshaper
import linAlg
#
def mapToUnit(x,xBound):
    """
    Linearly map x\in[xBound] to xi\in[-1,1]
    x can be either scalar or a vector
    """
    x = np.array(x, copy=False, ndmin=1)
    xi=(2.*(x[:]-xBound[0])/(xBound[1]-xBound[0])-1.)
    return xi
#
def mapFromUnit(xi,xBound):
    """
    Linearly map xi\in[-1,1] to x\in[xBound]
    x can be either scalar or a vector
    """
    xi = np.array(xi, copy=False, ndmin=1)
    x=(0.5*(xi[:]+1.0)*(xBound[1]-xBound[0])+xBound[0])
    return x
#
def gqPtsWts(n,type_):
    """
    Gauss quadrature points and weights of type type_    
    """
    if type_=='Unif':
       x=np.polynomial.legendre.leggauss(n)
    elif type_=='Norm':
       x=np.polynomial.hermite_e.hermegauss(n)
       #x=np.polynomial.hermite.hermgauss(n)
    quads=x[0]
    weights=x[1]
    return quads,weights
#
def map_xi2q(xi,xAux,distType):
    """
    Map xi\in\Gamma to q \in Q, where \Gamma is the space corresponding to standard gPCE
       xAux=xBound if distType='Unif', xi\in[-1,1]
       xAux=[m,v] if disType='Norm' where x~N(m,v), xi\in[-\inf,\inf]
    """   
    xi = np.array(xi, copy=False, ndmin=1)
    if distType=='Unif':
       x=mapFromUnit(xi,xAux)   
    elif distType=='Norm':
       x=xAux[0]+xAux[1]*xi
    return x
#
def pceBasis(n,xi,distType_):
    """
    Evaluate polynomial of order n at xi\in\Gamma
    The standard polynomials are choosen based on gPCE rules. 
    """
    if distType_=='Unif':
       v=np.polynomial.legendre.legval(xi,[0]*n+[1])
    elif distType_=='Norm':
       v=np.polynomial.hermite_e.hermeval(xi,[0]*n+[1])
    return v
#
def pceDensity(xi,distType_):
    """
    Evaluate the PDF of the standard (in gPCE sense) random variable of distribution
       type distType_ at points xi\in\Gamma (mapped space)
    """
    if distType_=='Unif':
       pdf_=0.5*np.ones(xi.shape[-1]) 
    elif distType_=='Norm':
       pdf_=np.exp(-0.5*xi**2.)/mt.sqrt(2*mt.pi)
    return pdf_   
#
def gqInteg_fac(distType):
    """
    Multipliers for the GQ rule given the weights provided by numpy
    """
    if distType=='Unif':
       fac_=0.5 
    elif distType=='Norm':
       fac_=1./mt.sqrt(2*mt.pi) 
    return fac_   
#
def pceDict_corrector(pceDict):
    """
       Correct pceDict for PCE to ensure consistency. 
         * For 'GQ' samples+'TP' truncation method: either 'Projection' or 'Regression' can be used
         * For all combination of sample points and truncation, 'Projection' can be used to compute PCE coefficients
    """
    #single-D parameter, p==1
    if 'truncMethod' not in pceDict:
       if pceDict['pceSolveMethod']=='Projection':
          if pceDict['sampleType'] !='GQ':
             pceDict['pceSolveMethod']='Regression'
             print("... Original 'Projection' method for PCE is replaced by 'Regression'.")
    else:   #multi-D parameter, p>1
       if pceDict['truncMethod']=='TO':
          if 'pceSolveMethod' not in pceDict or pceDict['pceSolveMethod']!='Regression':
             pceDict['pceSolveMethod']='Regression'
             print("... Original method for PCE is replaced by 'Regression'.")
       if pceDict['truncMethod']=='TP':
          if 'sampleType' not in pceDict or pceDict['sampleType']!='GQ':
             pceDict['pceSolveMethod']='Regression'
             print("... Original method for PCE is replaced by 'Regression'.")
       if pceDict['pceSolveMethod']=='Regression' and pceDict['truncMethod']!='TP':
          LMax_def=10   #default value of LMax
          if 'LMax' not in pceDict:
             print("WARNING in pceDict: 'LMax' should be set when Total-Order method is used.")
             print("Here 'LMax' is set to default value %d" %LMax_def)
             pceDict.update({'LMax':LMax_def})
    return pceDict
#
def PCE_coef_conv_plot(fCoef,kSet,distType,pwOpts):
    """
       Plot convergence of PCE terms
       ||fk*Psi_k||/||f0*Psi_0|| is plotted versus |k|=sum(k_i)
       Inputs:
          fCoef: 1D array of length K containing a PCE coefficients
          kSet: Index set of PCE, kSet=[[k1,k2,...,kp],...], if empty: 1d param space
          distType: A list containing distribution type of RVs, distType=[dist1,dist2,...,distp]      
          pwOpts: (optional) options to save the figure, keys: 'figDir', 'figName' 
    """
    K=len(fCoef)   #no of terms in PCE
    if not kSet:   #1d parameter space
       p=1
       kSet=[]
       for i in range(K):
           kSet.append([i])
    else:
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
            psi_k_=pceBasis(k_,xi,distType[ip])
            PsiNorm*=np.linalg.norm(psi_k_,2)
            if distType[ip] not in ['Unif','Norm']:
                print('...... ERROR in PCE_coef_conv_plot(): distribution %s is not available!'%distType[ip])
        termNorm.append(abs(fCoef[ik])*PsiNorm)
    termNorm0=termNorm[0]
    #plot
    plt.figure(figsize=(10,5))
    plt.semilogy(kMag,termNorm/termNorm0,'ob',fillstyle='none')
    plt.xlabel(r'$|\mathbf{k}|$',fontsize=18)
    plt.ylabel(r'$|\hat{f}_\mathbf{k}|\, ||\Psi_{\mathbf{k}(\mathbf{\xi})}||_2/|\hat{f}_0|$',fontsize=18)
    plt.xticks(ticks=kMag,fontsize=17)
    plt.yticks(fontsize=17)
    plt.grid(alpha=0.3)
    if pwOpts:
       if 'ylim' in pwOpts: 
           plt.ylim(pwOpts['ylim']) 
       fig = plt.gcf()
       DPI = fig.get_dpi()
       fig.set_size_inches(800/float(DPI),400/float(DPI))
       figDir=pwOpts['figDir']
       if not os.path.exists(figDir):
          os.makedirs(figDir)
       outName=pwOpts['figName']
       plt.savefig(figDir+outName+'.pdf',bbox_inches='tight')   
       plt.show()
    else:   
       plt.show()
#
def pce_1d_cnstrct(fVal,xi,pceDict):
    """ 
       Construct a PCE over a 1D parameter space. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) , K=truncation of the sum
       Input:
           fVal: 1d numpy array containing response value at the samples
           xi: A 1D numpy array of parameter samples over [-1,1] space
               NOTE: Always Required to be provided unless pceDict='GQ' samples are used
           pceDict: A dictionary containing different options for PCE with these keys:   
               'distType': distribution type of the parameter
               'pceSolveMethod': method of solving for PCE coeffcients
                                 ='Projection' (Requires samples to be Gauss-Quadratures)
                                 ='Regression' (For uniquely-, over-, and under-determined systems. In the latter compressed sensing with L1/L2 regularization is automatically applied.)                                                  
               'sampleType': type of parameter samples at which observations are made
                                 ='GQ' (Gauss Quadrature nodes)
                                 =' '  (Any other nodal set)
          Output:
             fCoef: Coefficients in the PCE, length =K
             fMean: PCE estimation for E[f(q)]
             fVar:  PCE estimation for V[f(q)]
    """
    sampleType=pceDict['sampleType']         #Types of parameter samples
    pceSolveMethod=pceDict['pceSolveMethod'] #Method of solving for PCe coeffcients
    distType=pceDict['distType']             #Type of distribution of the parameter
    if sampleType=='GQ' and pceSolveMethod=='Projection':
       fCoef,fMean,fVar=pce_1d_GQ_cnstrct(fVal,distType) 
    else:   #Regression method
       if 'LMax' in pceDict.keys(): 
          LMax=pceDict['LMax']     
       else:
          LMax=len(fVal) 
          print("...... No 'LMax' existed, so 'LMax=nQ'")
       fCoef,fMean,fVar=pce_1d_nonGQ_cnstrct(fVal,xi,LMax,distType) 
    return fCoef,fMean,fVar
#
def pce_1d_GQ_cnstrct(fVal,distType):
    """ 
       Construct a PCE over a 1D parameter space. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Using Projection method with Gauss-Quadrature method to compute the PCE coefficients
       K=number of GQ samples = len(fVal)
       distType: distribution type of the parameter
    """
    nQ=len(fVal) #number of quadratures (collocation samples)
    xi,w=gqPtsWts(nQ,distType)
    K=nQ  #upper bound of sum in PCE
    #find the coefficients in the expansion
    fCoef=np.zeros(nQ)
    sum2=[]
    fac_=gqInteg_fac(distType)
    for k in range(K):  #k-th coeff in PCE
        psi_k=pceBasis(k,xi,distType)
        sum1=np.sum(fVal[:K]*psi_k[:K]*w[:K]*fac_)
        sum2_=np.sum((psi_k[:K])**2.*w[:K]*fac_)
        fCoef[k]=sum1/sum2_
        sum2.append(sum2_)
    #find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=np.sum(fCoef[1:]**2.*sum2[1:])
    return fCoef,fMean,fVar
#
def pce_1d_nonGQ_cnstrct(fVal,xi,LMax,distType):
    """ 
       Construct a PCE over a 1D parameter space. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Using Regression method with arbitrary K=LMax to compute PCE coefficients
       K=LMax
       Note: The linear system in Regression method can be determined, over-/under-determined. 
             In the latter, compressed sensing is automatically applied.
       distType: distribution type of the parameter
    """     
    nQ=len(fVal) #number of quadratures (collocation samples)
    K=LMax      #truncation in the PCE
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    #(2) Find the coefficients in the expansion:Only Regression method can be used. 
    #    Also we need to compute gamma_k (=sum2) required to estimate the variance by PCE.
    #    For this we create an auxiliary Gauss-Quadrature grid to compute intgerals
    A=np.zeros((nData,K))    #Matrix of known coeffcient for regression to compute PCE coeffcients
    sum2=[]
    xi_aux,w_aux=gqPtsWts(K+1,distType)  #auxiliary GQ rule for computing gamma_k
    fac_=gqInteg_fac(distType)
    for k in range(K):    
        psi_aux=pceBasis(k,xi_aux,distType)
        for j in range(nData):
            A[j,k]=pceBasis(k,xi[j],distType)
        sum2.append(np.sum((psi_aux[:K+1])**2.*w_aux[:K+1]*fac_))
    #Find the PCE coeffs by Linear Regression 
    fCoef=linAlg.myLinearRegress(A,fVal)   #This can be over-, under-, or uniquely- determined systetm.
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=np.sum(fCoef[1:]**2.*sum2[1:])
    return fCoef,fMean,fVar
#
def pce_1d_eval(fk,xi,distType):
    """ 
       Evaluate a 1D PCE at a set of test points xi\in[-1,1]
       Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
       distType: distribution type of the parameter
    """
    K=len(fk) 
    xi = np.array(xi, copy=False, ndmin=1)
    fpce=[]
    for i in range(xi.size):
        sum1=0.0;
        for k in range(K):
            sum1+=fk[k]*pceBasis(k,xi[i],distType)
        fpce.append(sum1)
    return fpce
#
def pce_2d_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 2D parameter space. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) , K=truncation of the sum
          Input:
             fVal: 1d numpy array containing response value at the samples
             nQList: A list containing number of samples, nQList=[nQ1,nQ2]
                     For Tensor Product method: nQ1,nQ2 are number of samples in each direction
                     For Other methods: It is not used in construction of PCE
             xiGrid: A 2D grid of sampled parameters mapped in [-1,1]^2 space
                     xiGrid=2d np array, [nData,2]
                     Always Required to be provided unless 'GQ' samples are used
             pceDict: A dictionary containing different options for PCE with these keys:   
                    'distType': distribution type of the parameter
                    'truncMethod': method of truncating PCE
                                 ='TP' (tensor product method)         
                                 ='TO' (total order method)
                    'pceSolveMethod': method of solving for PCE coeffcients
                                 ='Projection' (Requires samples to be Gauss-Quadratures)
                                 ='Regression' (For uniquely-, over-, and under-determined systems. In the latter compressed sensing with L1/L2 regularization is automatically applied.)                                                  
                                 NOTE: For 'GQ'+'TP', the pceSolveMethod is 'Projection'. For any other combination, we use 'Regression'
                    'sampleType': type of parameter samples at which observations are made
                                ='GQ' (Gauss Quadrature nodes)
                                =' '  (Any other nodal set)
          Output:
             fCoef: Coefficients in the PCE, length =K
             kSet:  Index set, list of p-d lists [[k1,1,k2,1],...,[k1,K,k2,K]]
             fMean: PCE estimation for E[f(q)]
             fVar:  PCE estimation for V[f(q)]
    """
    truncMethod=pceDict['truncMethod']   #Truncation method for PCE
    sampleType=pceDict['sampleType']     #Types of parameter samples
    if sampleType=='GQ' and truncMethod=='TP':   #Gauss Quadrature samples with Tensor-Product Rules (use either Projection or Regression)
       fCoef,kSet,fMean,fVar=pce_2d_GQTP_cnstrct(fVal,nQList,pceDict)
    else:                  #Any other type of samples (structured/unstructured)+Regression method
       fCoef,kSet,fMean,fVar=pce_2d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict)
    return fCoef,kSet,fMean,fVar
#
def pce_2d_GQTP_cnstrct(fVal,nQList,pceDict):
    """ 
       Construct a PCE over a 2D parameter space 
       * Type of parameter samples: Gauss-Quadrature nodes
       * Method of truncating PCE: Tensor Product Method
       * pceSolveMethod= 'Projection' or 'Regression 
       * Only case where Projection can be used 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Input:
            fVal: 1d numpy array containing response value at the samples
            nQList: List of number of GQ nodes in both directions, [nQ1,nQ2]
            pceDict: A dictionary containing different options for PCE 
       Output:
            fCoef: Coefficients in the PCE: length =K
            kSet:  Index set, list of p-d lists [[k1,1,k2,1],...,[k1,K,k2,K]]
            fMean: PCE estimation for E[f(q)]
            fVar:  PCE estimation for V[f(q)]
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    print('... A gPCE for a 2D parameter space is constructed.')
    print('...... Samples in each direction are Gauss Quadrature nodes (User should check this!).')
    print('...... PCE truncation method: TP')    
    print('...... Method of computing PCE coefficients: %s' %pceDict['pceSolveMethod'])
    distType=pceDict['distType']
    #(1) Set variables
    nQ1=nQList[0]
    nQ2=nQList[1]
    xi1,w1=gqPtsWts(nQ1,distType[0])
    xi2,w2=gqPtsWts(nQ2,distType[1])
    K=nQ1*nQ2;  #upper bound of sum in PCE using Tensor Product truncation
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    if K!=nData:
       print('ERROR in pce_2d_GQTP_cnstrct(): ')
    #(2) Find the coefficients in the expansion
    #By default, Projection method is used (assuming samples are Gauss-Quadrature points)
    fCoef=np.zeros(K)
    sum2=[]
    kSet=[]    #index set
    for k2 in range(nQ2):  #k-th coeff in PCE - param2
        psi_k2=pceBasis(k2,xi2,distType[1])
        for k1 in range(nQ1):  #k-th coeff in PCE - param1
            psi_k1=pceBasis(k1,xi1,distType[0])
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

    print('here',fVal)        
    #(2b) Recompute fCoef via Regression, in case Regression is chosen
    if pceDict['pceSolveMethod']=='Regression':
       xiGrid=reshaper.vecs2grid(xi1,xi2)
       A=np.zeros((nData,K))
       k=-1
       for k2 in range(nQ2): 
           for k1 in range(nQ1): 
               k+=1
               for j in range(nData):
                   A[j,k]=pceBasis(k1,xiGrid[j,0],distType[0])*pceBasis(k2,xiGrid[j,1],distType[1])
       #Linear Regression to solve the linear set of equations
       fCoef=linAlg.myLinearRegress(A,fVal)   #This is a uniquely determined system
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**2.)*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,kSet,fMean,fVar
#
def pce_2d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 2D parameter space 
       * Method of truncating PCE: Total Order or Tensor Product
       * Method of solving for PCE coefficients: ONLY Regression
       * Use this routine for any combination but 'GQ'+'TP'
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Input:
           fVal: 1d numpy array containing response value at the samples
           xiGrid: sampled parameters mapped to [-1,1]^2  
                  A Grid of nData x 2
                  NOTE: xiGrid is ALWAYS needed in this routine
           nQList: Used only if 'TP' is chosen, [nQ1,nQ2]
           pceDict: A dictionary containing different options for PCE 
       Output:
            fCoef: Coefficients in the PCE: length =K
            kSet:  Index set, list of 2d lists [[k1,1,k2,1],...,[k1,K,k2,K]]
            fMean: PCE estimation for E[f(q)]
            fVar:  PCE estimation for V[f(q)]
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    pceSolveMethod=pceDict['pceSolveMethod']
    truncMethod=pceDict['truncMethod']
    distType=pceDict['distType']
    print('... A gPCE for a 2D parameter space is constructed.')
    print('...... PCE truncation method: %s' %truncMethod)    
    print('...... Method of computing PCE coefficients: %s' %pceSolveMethod)
    if pceDict['truncMethod']=='TO':
       LMax=pceDict['LMax']   #max order of polynomial in each direction
       print('         with LMax=%d as the max polynomial order in each direction.' %LMax)
    if pceSolveMethod!='Regression':
       print('ERROR in pce_2d_nonGPTP_cnstrct(): only Regression method can be used for PCE with Total Order truncation method.')

    #(1) Set variables
    #Number of terms in PCE
    if truncMethod=='TO':
       K=int((LMax+2)*(LMax+1)/2) 
       N1=LMax+1  #loops' upper bound
       N2=LMax+1
    elif truncMethod=='TP':   #'TP' needs nQList 
       N1=nQList[0]
       N2=nQList[1]
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
    xi1_aux,w1_aux=gqPtsWts(N1,distType[0])  #auxiliary GQ rule for computing gamma_k
    xi2_aux,w2_aux=gqPtsWts(N2,distType[1])  
    for k2 in range(N2):    
        psi_aux_k2=pceBasis(k2,xi2_aux,distType[1])
        for k1 in range(N1):
            psi_aux_k1=pceBasis(k1,xi1_aux,distType[0])
            if (truncMethod=='TO' and (k1+k2)<=LMax) or (truncMethod=='TP'):
               #constructing Aij
               k+=1
               kSet.append([k1,k2])    #index set
               for j in range(nData):
                   A[j,k]=pceBasis(k1,xiGrid[j,0],distType[0])*pceBasis(k2,xiGrid[j,1],distType[1])
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
#
def pce_2d_eval(fk,kSet,xi1,xi2,distType):
    """ 
       Evaluate a PCE over a 2D param space at a set of test points xi1,xi2\in[-1,1] 
       i.e., Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
       NOTE: This routine works for any PCE no matter what truncation method has been used for its construction. Because, we use the index set kSet.
         Input:
            fk: 1D array of length K containing the coefficients of the PCE
            kSet: List of indices:[[k1,1,k2,1],[k1,2,k2,2],...,[k1,K,k2,K]] produced based on the tensor product or total order rules when constructing the PCE
            xi1,xi2: Test points in each direction of the 2D parameter space. Always a tensor product grid is constructed based on test points to evaluate the PCE.
            distType: List of distribution type of the parameters
         Output: 
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
                sum1+=fk[k]*pceBasis(k1,xi1[i1],distType[0])*pceBasis(k2,xi2[i2],distType[1])
            fpce[i1,i2]=sum1
    return fpce
#
def pce_3d_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 3D parameter space. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) , K=truncation of the sum
          Input:
             fVal: 1d numpy array containing response value at the samples
             nQList: A list containing number of samples, nQList=[nQ1,nQ2,nQ3]
                     For Tensor Product method: nQ1,nQ2,nQ3 are number of samples in each direction
                     For Other methods: It is not used in construction of PCE
             xiGrid: A 3D grid of sampled parameters mapped in [-1,1]^3 space
                     xiGrid=2d np array, [nData,3]
                     Always Required to be provided unless 'GQ' samples are used
             pceDict: A dictionary containing different options for PCE with these keys:   
                    'distType': distribution type of the parameter
                    'truncMethod': method of truncating PCE
                                 ='TP' (tensor product method)         
                                 ='TO' (total order method)
                    'pceSolveMethod': method of solving for PCE coeffcients
                                 ='Projection' (Requires samples to be Gauss-Quadratures)
                                 ='Regression' (For uniquely-, over-, and under-determined systems. In the latter compressed sensing with L1/L2 regularization is needed.)                                                  
                                 NOTE: For 'GQ'+'TP', the pceSolveMethod is 'Projection'. For any other combination, we use 'Regression'
                    'sampleType': type of parameter samples at which observations are made
                                ='GQ' (Gauss Quadrature nodes)
                                =' '  (Any other nodal set)
          Output:
             fCoef: Coefficients in the PCE: length =K
             kSet:  Index set, list of p-d lists [[k1,1,k2,1,k3,1],...,[k1,K,k2,K,k3,K]]
             fMean: PCE estimation for E[f(q)]
             fVar:  PCE estimation for V[f(q)]
    """
    truncMethod=pceDict['truncMethod']   #Truncation method for PCE
    sampleType=pceDict['sampleType']     #Types of parameter samples
    if sampleType=='GQ' and truncMethod=='TP':   #Gauss Quadrature samples with Tensor-Product Rules (use either Projection or Regression)
       fCoef,kSet,fMean,fVar=pce_3d_GQTP_cnstrct(fVal,nQList,pceDict)
    else:                  #Any other type of samples (structured/unstructured)+Regression method
       fCoef,kSet,fMean,fVar=pce_3d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict)
    return fCoef,kSet,fMean,fVar
#
def pce_3d_GQTP_cnstrct(fVal,nQList,pceDict):
    """ 
       Construct a PCE over a 3D parameter space 
       * Type of parameter samples: Gauss-Quadrature nodes
       * Method of truncating PCE: Tensor Product Method
       * pceSolveMethod= 'Projection' or 'Regression 
       * Only case where Projection can be used 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Input:
            fVal: 1d numpy array containing response value at the samples
            nQList: List of number of GQ nodes in both directions, [nQ1,nQ2,nQ3]
            pceDict: A dictionary containing different options for PCE 
       Output:
            fCoef: Coefficients in the PCE: length =K
            kSet:  Index set, list of p-d lists [[k1,1,k2,1,k3,1],...,[k1,K,k2,K,k3,K]]
            fMean: PCE estimation for E[f(q)]
            fVar:  PCE estimation for V[f(q)]
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    print('... A gPCE for a 3D parameter space is constructed.')
    print('      * Samples in each direction are Gauss Quadrature nodes (User should check this!).')
    print('      * PCE truncation method: TP')    
    print('      * Method of computing PCE coefficients: %s' %pceDict['pceSolveMethod'])
    distType=pceDict['distType']
    #(1) Set variables
    nQ1=nQList[0]
    nQ2=nQList[1]
    nQ3=nQList[2]
    xi1,w1=gqPtsWts(nQ1,distType[0])
    xi2,w2=gqPtsWts(nQ2,distType[1])
    xi3,w3=gqPtsWts(nQ3,distType[2])
    K=nQ1*nQ2*nQ3    #upper bound of sum in PCE using Tensor Product truncation
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    if K!=nData:
       print('ERROR in pce_3d_GQTP_cnstrct(): ')
    #(2) Find the coefficients in the expansion
    #By default, Projection method is used (assuming samples are Gauss-Quadrature points)
    fCoef=np.zeros(K)
    sum2=[]
    kSet=[]   #index set
    for k3 in range(nQ3):      #k-th coeff in PCE - param3
        psi_k3=pceBasis(k3,xi3,distType[2])
        for k2 in range(nQ2):  #k-th coeff in PCE - param2
            psi_k2=pceBasis(k2,xi2,distType[1])
            for k1 in range(nQ1):  #k-th coeff in PCE - param1
                psi_k1=pceBasis(k1,xi1,distType[0])
                sum1=0.0
                sum2_=0.0
                k=(k3*nQ2*nQ1)+(k2*nQ1)+k1
                kSet.append([k1,k2,k3])
                for j3 in range(nQ3):
                    for j2 in range(nQ2):
                        for j1 in range(nQ1):
                            j=(j3*nQ2*nQ1)+(j2*nQ1)+j1
                            sum1 += fVal[j]*(psi_k1[j1]*psi_k2[j2]*psi_k3[j3]*w1[j1]*w2[j2]*w3[j3])
                            sum2_+=         (psi_k1[j1]*psi_k2[j2]*psi_k3[j3])**2.*w1[j1]*w2[j2]*w3[j3]
                fCoef[k]=sum1/sum2_
                sum2.append(sum2_)
    #(2b) Recompute fCoef via Regression, in case Regression is chosen
    if pceDict['pceSolveMethod']=='Regression':
       xiGrid=reshaper.vecs2grid3d(xi1,xi2,xi3)
       A=np.zeros((nData,K))
       k=-1
       for k3 in range(nQ3): 
           for k2 in range(nQ2): 
               for k1 in range(nQ1):
                   k+=1
                   for j in range(nData):
                       A[j,k]=pceBasis(k1,xiGrid[j,0],distType[0])*pceBasis(k2,xiGrid[j,1],distType[1])*pceBasis(k3,xiGrid[j,2],distType[2])
       #Linear Regression to solve the linear set of equations
       fCoef=linAlg.myLinearRegress(A,fVal)   #This is a uniquely determined system
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**3.)*fCoef[k]*fCoef[k]*sum2[k]    #0.5:PDF of U on [-1,1]
    return fCoef,kSet,fMean,fVar
#
def pce_3d_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a 3D parameter space 
       * Method of truncating PCE: Total Order or Tensor Product
       * Method of solving for PCE coefficients: ONLY Regression
       * Use this routine for any combination but 'GQ'+'TP'
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) 
       Input:
           fVal: 1d numpy array containing response value at the samples
           xiGrid: sampled parameters mapped to [-1,1]^3
                  A Grid of nData x 2
                  NOTE: xiGrid is ALWAYS needed in this routine
           nQList: Used only if 'TP' is chosen, [nQ1,nQ2,nQ3]
           pceDict: A dictionary containing different options for PCE 
       Output:
            fCoef: Coefficients in the PCE: length =K
            kSet:  Index set, list of 3d lists [[k1,1,k2,1,k3,1],...,[k1,K,k2,K,k3,K]]
            fMean: PCE estimation for E[f(q)]
            fVar:  PCE estimation for V[f(q)]
    """
#    print('NOTE: Make sure the reponses are imported by keeping the loop of the 2nd param outside of that of param1 (Always: latest parameter has the outermost loop)')
    pceSolveMethod=pceDict['pceSolveMethod']
    truncMethod=pceDict['truncMethod']
    distType=pceDict['distType']
    print('... A gPCE for a 3D parameter space is constructed.')
    print('...... PCE truncation method: %s' %truncMethod)    
    print('...... Method of computing PCE coefficients: %s' %pceSolveMethod)
    if pceDict['truncMethod']=='TO':
       LMax=pceDict['LMax']   #max order of polynomial in each direction
       print('         with LMax=%d as the max polynomial order in each direction.' %LMax)
    if pceSolveMethod!='Regression':
       print('ERROR in pce_3d_nonGPTP_cnstrct(): only Regression method can be used for PCE with Total Order truncation method.')

    #(1) Set variables
    #Number of terms in PCE
    if truncMethod=='TO':
       K=int((LMax+3)*(LMax+2)*(LMax+1)/6)  
       N1=LMax+1  #loops' upper bound
       N2=LMax+1
       N3=LMax+1
    elif truncMethod=='TP':   #'TP' needs nQList 
       N1=nQList[0]
       N2=nQList[1]
       N3=nQList[2]
       K=N1*N2*N3           
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
    xi1_aux,w1_aux=gqPtsWts(N1,distType[0])  #auxiliary GQ rule for computing gamma_k
    xi2_aux,w2_aux=gqPtsWts(N2,distType[1])
    xi3_aux,w3_aux=gqPtsWts(N3,distType[2])
    for k3 in range(N3):    
        psi_aux_k3=pceBasis(k3,xi3_aux,distType[2])
        for k2 in range(N2):    
            psi_aux_k2=pceBasis(k2,xi2_aux,distType[1])
            for k1 in range(N1):
                psi_aux_k1=pceBasis(k1,xi1_aux,distType[0])
                if (truncMethod=='TO' and (k1+k2+k3)<=LMax) or (truncMethod=='TP'):
                   #constructing Aij
                   k+=1
                   kSet.append([k1,k2,k3])    #index set
                   for j in range(nData):
                       A[j,k]=pceBasis(k1,xiGrid[j,0],distType[0])*pceBasis(k2,xiGrid[j,1],distType[1])*pceBasis(k3,xiGrid[j,2],distType[2])
                   #computing sum2=gamma_k
                   sum2_=0.0
                   k_aux=(k3*N2*N1)+(k2*N1)+k1
                   for j3 in range(N3):
                       for j2 in range(N2):
                           for j1 in range(N1):
                               sum2_+=(psi_aux_k1[j1]*psi_aux_k2[j2]*psi_aux_k3[j3])**2.*w1_aux[j1]*w2_aux[j2]*w3_aux[j3]
                   sum2.append(sum2_)
    #Find the PCE coeffs by Linear Regression 
    fCoef=linAlg.myLinearRegress(A,fVal)   #This can be over-, under-, or uniquely- determined systetm.
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=(0.5**3.)*fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,kSet,fMean,fVar
#
def pce_3d_eval(fk,kSet,xi1,xi2,xi3,distType):
    """ 
       Evaluate a PCE over a 3D param space at a set of test points xi1,xi2,xi3\in[-1,1] 
       i.e., Given {f_k}, find f(q)=\sum_k f_k psi_k(q) 
       NOTE: This routine works for any PCE no matter what truncation method has been used for its construction. Because, we use the index set kSet.
       Uniform Uncertain Parameter    => Legendre Polynomials
         Input:
            fk: 1D array of length K containing the coefficients of the PCE
            kSet: List of indices:[[k1,1,k2,1,k3,1],[k1,2,k2,2,k3,2],...,[k1,K,k2,K,k3,K]] produced based on the tensor product or total order rules when constructing the PCE
            xi1,xi2,xi3: Test points in each direction of the 3D parameter space. Always a tensor product grid is constructed based on test points to evaluate the PCE.
            distType: List of distribution type of the parameters
         Output: 
           fpce: Response values predicted (inerpolated) by the PCE at the test points
    """
    xi1 = np.array(xi1, copy=False, ndmin=1)
    xi2 = np.array(xi2, copy=False, ndmin=1)
    xi3 = np.array(xi3, copy=False, ndmin=1)
    n1=xi1.size
    n2=xi2.size
    n3=xi3.size
    fpce=np.zeros((n1,n2,n3))
    K=len(kSet)   #number of terms in the PCE
    for i3 in range(n3):
        for i2 in range(n2):
            for i1 in range(n1):
                sum1=0.0
                for k in range(K):
                    k1=kSet[k][0]
                    k2=kSet[k][1]
                    k3=kSet[k][2]
                    sum1+=fk[k]*pceBasis(k1,xi1[i1],distType[0])*pceBasis(k2,xi2[i2],distType[1])*pceBasis(k3,xi3[i3],distType[2])
                fpce[i1,i2,i3]=sum1
    return fpce
#    
################################
# Tests
################################ 
#--------------------------------------------------
#To run a test:
#python -c 'import pce as X;X.pce_test_1d()'
#--------------------------------------------------
def pce_1d_test():
    """
    Test PCE for 1 uncertain parameter uniformly distributed over [a1,b1] using Legendre polynomial bases
    """
    print('------ gPCE TEST 1 ------')
    #--- settings -------------------------
    #Parameter settings
    distType='Norm'   #distribution type of the parameter
    if distType=='Unif':
       qBound=[-2,4.0]   #parameter range only if 'Unif'
    elif distType=='Norm':
       qAux=[1.,1.5]   #[m,v] for 'Norm' q~N(m,v^2)
    n=7   #number of training samples
    nTest=200   #number of test sample sin the parameter space
    #PCE Options
    sampleType='GQ'    #'GQ'=Gauss Quadrature nodes
                       #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Projection' #'Regression': for any combination of sample points 
                                #'Projection': only for GQ
    LMax_=10   #(Only needed for Regresson method), =K: truncation (num of terms) in PCE                               #(LMax will be over written by nSamples if it is provided for 'GQ'+'Projection')
              #NOTE: LMAX>=nSamples
    #--------------------------------------
    #(0) make the pceDict
    pceDict={'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,'LMax':LMax_,
             'distType':distType}
    pceDict=pceDict_corrector(pceDict)
    #
    #(1) generate training data
    if sampleType=='GQ':
       xi,w=gqPtsWts(n,distType)   #Gauss quadratures
    else:         
       #xi=2.*np.random.rand(n)-1.0 #random samples in [-1,1]       
       xi=np.linspace(-1,1,n)      #uniformly-spaced samples over [-1,1]
    
    if distType=='Unif':
       qAux=qBound
       fType='type1'
       q=map_xi2q(xi,qBound,distType)
       f=analyticTestFuncs.fEx1D(q,fType)   #function value at the parameter samples (Gauss quads)
    elif distType=='Norm':
       q=map_xi2q(xi,qAux,distType)
       qBound=[min(q),max(q)]
       fType='type2'
       f=analyticTestFuncs.fEx1D(q,fType,qAux)   #function value at the parameter samples (Gauss quads)
    #
    #(2) compute the exact moments (reference data)
    fMean_ex,fVar_ex=analyticTestFuncs.fEx1D_moments(qAux,fType)
    #
    #(3) construct the PCE
    fCoef,fMean,fVar=pce_1d_cnstrct(f,xi,pceDict)  #find PCE coefficients
    #
    #(4) compare moments: exact vs. PCE estimations
    print(writeUQ.printRepeated('-',70))
    print('-------------- Exact -------- PCE --------- Error % ') 
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_ex,fMean,(fMean-fMean_ex)/fMean_ex*100.))
    print('Var  of f(q) = %g\t%g\t%g' %(fVar_ex,fVar,(fVar-fVar_ex)/fVar_ex*100.))
    print(writeUQ.printRepeated('-',70))
    #
    #(5) plot convergence of PCE
    PCE_coef_conv_plot(fCoef,[],[distType],{})
    #
    #(6) plot
    # test samples
    qTest=np.linspace(qBound[0],qBound[1],nTest)  #test points in param space
    if distType=='Unif':
       xiTest=mapToUnit(qTest,qBound)
       fTest=analyticTestFuncs.fEx1D(qTest,fType)   #exact response at test points
    elif distType=='Norm':   
       xiTest=(qTest-qAux[0])/qAux[1]
       fTest=analyticTestFuncs.fEx1D(qTest,fType,qAux)   #exact response at test points

    fPCE=pce_1d_eval(fCoef,xiTest,distType)  #Prediction by PCE
    plt.figure(figsize=(12,5))
    ax=plt.gca()
    plt.plot(qTest,fTest,'-k',lw=2,label=r'Exact $f(q)$')
    plt.plot(q,f,'ob',label=sampleType+' Training Samples')
    plt.plot(qTest,fPCE,'-r',lw=2,label='PCE')
    plt.plot(qTest,fMean*np.ones(len(qTest)),'-b',label=r'$\mathbb{E}(f(q))$') 
    ax.fill_between(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),color='powderblue',alpha=0.4)
    plt.plot(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b',label=r'$\mathbb{E}(f(q))\pm 95\%CI$')
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
    Test PCE for 2 uncertain parameters uniformly distributed over [a1,b1]x[a2,b2] using Legendre polynomial bases
    """
    print('------ gPCE TEST 2 ------')
    #settings------------
    distType=['Unif','Unif']   #distribution type of the parameters
    qBound=[[-1,2],   #parameter range 
            [-3,3]] 
    nQ=[7,5]   #number of collocation smaples of param1
    nTest=[100,101]   #number of test points in parameter spaces
    #PCE Options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType=''     #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=8   #max polynomial order in each parameter direction
    #--------------------

    #generate observations   
    xi1,w1=gqPtsWts(nQ[0],'Unif')   #Gauss sample pts in [-1,1]
    xi2,w2=gqPtsWts(nQ[1],'Unif')   #Gauss sample pts in [-1,1]
    q1=mapFromUnit(xi1,qBound[0])    #map Gauss points to param1 space
    q2=mapFromUnit(xi2,qBound[1])    #map Gauss points to param2 space
    fVal=analyticTestFuncs.fEx2D(q1,q2,'type1','tensorProd')  #function value at the parameter samples (Gauss quads)    
    #construct the gPCE
    pceDict={'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    pceDict=pceDict_corrector(pceDict)
    xiGrid=reshaper.vecs2grid(xi1,xi2)
    fCoefs,kSet,fMean,fVar=pce_2d_cnstrct(fVal,nQ,xiGrid,pceDict)

    #plot convergence of PCE terms
    PCE_coef_conv_plot(fCoefs,kSet,distType,{})

    #make predictions at test points in the parameter space
    q1Test =np.linspace(qBound[0][0],qBound[0][1],nTest[0])  #test points in param1 space
    xi1Test=mapToUnit(q1Test,qBound[0])
    q2Test =np.linspace(qBound[1][0],qBound[1][1],nTest[1])  #test points in param2 space
    xi2Test=mapToUnit(q2Test,qBound[1])
    fTest=analyticTestFuncs.fEx2D(q1Test,q2Test,'type1','tensorProd')   #response value at the test points
    fPCE=pce_2d_eval(fCoefs,kSet,xi1Test,xi2Test,distType)  #Prediction at test points by PCE
    #create 2D grid and response surface over it
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F')
    fErrorGrid=np.zeros((nTest[0],nTest[1]))
    for j in range(nTest[1]):
        for i in range(nTest[0]):
            k=i+j*nTest[0]
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
    CS1 = plt.contour(q1Test,q2Test,fTestGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact Response')

    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(q1Test,q2Test,fPCE.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Surrogate Response')

    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(q1Test,q2Test,fErrorGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.0f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.title('|Exact-Surrogate|/Exact %')
    plt.show()
#     
def pce_3d_test():
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
    #PCE options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'   #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=9   #max polynomial order in each parameter direction
    #--------------------
    #generate observations   
    xi1,w1=gqPtsWts(nQ1,'Unif')   #Gauss sample pts in [-1,1]
    xi2,w2=gqPtsWts(nQ2,'Unif')   #Gauss sample pts in [-1,1]
    xi3,w3=gqPtsWts(nQ3,'Unif')   #Gauss sample pts in [-1,1]
    q1=mapFromUnit(xi1,q1Bound)    #map Gauss points to param1 space
    q2=mapFromUnit(xi2,q2Bound)    #map Gauss points to param2 space
    q3=mapFromUnit(xi3,q3Bound)    #map Gauss points to param3 space
    fVal=analyticTestFuncs.fEx3D(q1,q2,q3,'Ishigami','tensorProd',funOpt)  #function value at the parameter samples (Gauss quads)    
    #construct the gPCE and compute moments
    pceDict={'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    pceDict=pceDict_corrector(pceDict)
    xiGrid=reshaper.vecs2grid3d(xi1,xi2,xi3)
    fCoefs,kSet,fMean,fVar=pce_3d_cnstrct(fVal,[nQ1,nQ2,nQ3],xiGrid,pceDict)
    #plot convergence of PCE terms
    PCE_coef_conv_plot(fCoefs,kSet,['Unif','Unif','Unif'],{})
    #exact moments of Ishigami function
    m,v=analyticTestFuncs.ishigami_exactMoments(q1Bound,q2Bound,q3Bound,funOpt)
    #print the results
    print(writeUQ.printRepeated('-',50))
    print('\t\t Exact \t\t PCE')
    print('E[f]:  ',m,fMean)
    print('V[f]:  ',v,fVar)
