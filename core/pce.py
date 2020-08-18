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
    Inputs:
       fVal: 1d numpy array containing response value at n training samples
       xi: A 1D numpy array of parameter samples over the mapped space
           NOTE: Always had to be provided unless 'sampleType'='GQ' samples are used
       pceDict: A dictionary containing different options for PCE with these keys:   
          'distType': distribution type of the parameter
          'pceSolveMethod': method of solving for PCE coeffcients
                   ='Projection' (Requires samples to be Gauss-Quadratures)
                   ='Regression' (For uniquely-, over-, and under-determined systems.)
                    In the latter compressed sensing with L1/L2 regularization is automatically applied.          'sampleType': type of parameter samples at which observations are made
                   ='GQ' (Gauss Quadrature nodes)
                   =' '  (Any other nodal set)
    Outputs:
       fCoef: Coefficients in the PCE, length =K
       fMean: PCE estimation for E[f(q)]
       fVar:  PCE estimation for V[f(q)]
    """
    sampleType=pceDict['sampleType']         
    pceSolveMethod=pceDict['pceSolveMethod'] 
    distType=pceDict['distType']             
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
    Construct a PCE over a 1D parameter space using Projection method with Gauss-quadrature 
        projection method to compute the PCE coefficients.
    Inputs:    
       fVal: 1d numpy array containing response value at n training samples
       distType: distribution type of the parameter
    Outputs:
       fCoef: Coefficients in the PCE, length =K
       fMean: PCE estimation for E[f(q)]
       fVar:  PCE estimation for V[f(q)]       
    """
    nQ=len(fVal) #number of quadratures (collocation samples)
    xi,w=gqPtsWts(nQ,distType)
    K=nQ  #upper bound of sum in PCE
    #Find the coefficients in the expansion
    fCoef=np.zeros(nQ)
    sum2=[]
    fac_=gqInteg_fac(distType)
    for k in range(K):  #k-th coeff in PCE
        psi_k=pceBasis(k,xi,distType)
        sum1=np.sum(fVal[:K]*psi_k[:K]*w[:K]*fac_)
        sum2_=np.sum((psi_k[:K])**2.*w[:K]*fac_)
        fCoef[k]=sum1/sum2_
        sum2.append(sum2_)
    #Estimate the mean and variance of f(q) 
    fMean=fCoef[0]
    fVar=np.sum(fCoef[1:]**2.*sum2[1:])
    return fCoef,fMean,fVar
#
def pce_1d_nonGQ_cnstrct(fVal,xi,LMax,distType):
    """ 
    Construct a PCE over a 1D parameter space using Regression method 
        with arbitrary K=LMax to compute PCE coefficients
    Inputs:    
       fVal: 1d numpy array containing response value at n training samples
       xi: 1d numpy array of parameter samples over the mapped space
       LMax: Max order in the PCE in each direction
       distType: distribution type of the parameter
    Outputs:
       fCoef: Coefficients in the PCE, length =K
       fMean: PCE estimation for E[f(q)]
       fVar:  PCE estimation for V[f(q)]       
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
    Evaluate a 1D PCE at a set of test points xi\in mapped space
    Inputs:
       fk: PCE coefficients, 1d numpy array
       xi: 1d numpy array of parameter samples over the mapped space
       distType: distribution type of the parameter
    Output:   
       fpce: PCE value at xi test samples
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
def pce_pd_cnstrct(fVal,nQList,xiGrid,pceDict):
    """ 
       Construct a PCE over a p-D parameter space, where p>1. 
       Find {f_k} in f(q)=\sum_k f_k psi_k(q) , K=truncation of the sum
          Input:
             fVal: 1d numpy array containing response value at n samples
             nQList: if 'TP': nQList=[nQ1,nQ2,...,nQp] where nQi: number of samples in i-th direction
                     if 'TO': nQlist=[]
             xiGrid: A pD grid of sampled parameters from the mapped space
                     a 2d numpy array of size [n,p]
                     Always had to be provided unless 'GQ' samples are used
             pceDict: A dictionary containing different options for PCE with these keys:   
                    'distType': distribution type of the parameter
                    'truncMethod': method of truncating PCE
                                 ='TP' (tensor product method)         
                                 ='TO' (total order method)
                    'pceSolveMethod': method of solving for PCE coeffcients
                                 ='Projection' (Requires samples to be Gauss-Quadratures)
                                 ='Regression' (For uniquely-, over-, and under-determined systems.)
                                  If under-determined compressed sensing with L1/L2 regularization 
                                  is automatically applied.          
                                  NOTE: For 'GQ'+'TP', the pceSolveMethod is always 'Projection'. 
                                        For any other combination, 'Regression' is used.
                    'sampleType': type of parameter samples at which observations are made
                                ='GQ' (Gauss quadrature nodes)
                                =' '  (Any other nodal set)
          Output:
             fCoef: Coefficients in the PCE, a 1d numpy array of size K
             kSet:  Index set, list of p-d lists [[k1,1,k2,1,...kp,1],...,[k1,K,k2,K,..,kp,K]]
             fMean: PCE estimation for E[f(q)]
             fVar:  PCE estimation for V[f(q)]
    """
    truncMethod=pceDict['truncMethod']   #Truncation method for PCE
    sampleType=pceDict['sampleType']     #Types of parameter samples
    if sampleType=='GQ' and truncMethod=='TP':   #Gauss Quadrature samples with Tensor-Product Rules (use either Projection or Regression)
       fCoef,kSet,fMean,fVar=pce_pd_GQTP_cnstrct(fVal,nQList,pceDict)
    else:                  #Any other type of samples (structured/unstructured)+Regression method
       fCoef,kSet,fMean,fVar=pce_pd_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict)
    return fCoef,kSet,fMean,fVar
#
def pce_pd_GQTP_cnstrct(fVal,nQList,pceDict):
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
    p=len(nQList)
    print('... A gPCE for a 2D parameter space is constructed.')
    print('...... Samples in each direction are Gauss Quadrature nodes (User should check this!).')
    print('...... PCE truncation method: TP')    
    print('...... Method of computing PCE coefficients: %s' %pceDict['pceSolveMethod'])
    distType=pceDict['distType']
    #(1) Quadrature rule
    xi=[]
    w=[]
    fac=[]
    K=1
    for i in range(p):
        xi_,w_=gqPtsWts(nQList[i],distType[i])
        xi.append(xi_)
        w.append(w_)
        K*=nQList[i]
        fac.append(gqInteg_fac(distType[i]))
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    if K!=nData:
       print('ERROR in pce_2d_GQTP_cnstrct(): ')
    #(2) Index set
    kSet=[]    #index set for the constructed PCE
    kGlob=np.arange(K)   #Global index
    kLoc=kGlob.reshape(nQList,order='F')  #Local index    
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
    fVal_=fVal.reshape(nQList,order='F').T #global to local index
    for k in range(K):
        psi_k=[]
        for j in range(p):
            psi_k.append(pceBasis(kSet[k][j],xi[j],distType[j]))
        sum1 =np.matmul(fVal_,(psi_k[0]*w[0]))*fac[0]
        sum2_=np.sum(psi_k[0]**2*w[0])*fac[0]
        for i in range(1,p):
            num_=(psi_k[i]*w[i])
            sum1=np.matmul(sum1,num_)*fac[i]
            sum2_*=np.sum(psi_k[i]**2*w[i])*fac[i]
        fCoef[k]=sum1/sum2_
        sum2.append(sum2_)
    #(3b) Compute fCoef via Regression
    if pceDict['pceSolveMethod']=='Regression':
       xiGrid=reshaper.vecs2grid(xi)
       A=np.zeros((nData,K))
       for k in range(K):
           aij_=pceBasis(kSet[k][0],xiGrid[:,0],distType[0])
           for i in range(1,p):
               aij_*=pceBasis(kSet[k][i],xiGrid[:,i],distType[i])
           A[:,k]=aij_
       fCoef=linAlg.myLinearRegress(A,fVal)   #This is a uniquely determined system
    #(4) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=np.sum(fCoef[1:]**2.*sum2[1:])
    return fCoef,kSet,fMean,fVar
#
def pce_pd_nonGQTP_cnstrct(fVal,nQList,xiGrid,pceDict):
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
    p=len(nQList)
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
    #(1) Preliminaries
    #Number of terms in PCE
    if truncMethod=='TO':
       K=int(mt.factorial(LMax+p)/(mt.factorial(LMax)*mt.factorial(p)))
       Nmax=(LMax+1)**p
       kOrderList=[LMax+1]*p
    elif truncMethod=='TP':   #'TP' needs nQList 
       K=np.prod(np.asarray(nQList))
       Nmax=K
       kOrderList=nQList
    # Quadrature rule to compute \gamma_k
    xi=[]
    w=[]
    fac=[]
    for i in range(p):
        xi_,w_=gqPtsWts(nQList[i],distType[i])
        xi.append(xi_)
        w.append(w_)
        fac.append(gqInteg_fac(distType[i]))
    # Index set
    kSet=[]    #index set for the constructed PCE
    kGlob=np.arange(Nmax)   #Global index
    kLoc=kGlob.reshape(kOrderList,order='F')  #Local index    
    for i in range(Nmax):
        k_=np.where(kLoc==kGlob[i])
        kSet_=[]
        for j in range(p):
            kSet_.append(k_[j][0])
        if (truncMethod=='TO' and sum(kSet_)<=LMax) or truncMethod=='TP':
           kSet.append(kSet_)
    print('...... Number of terms in PCE, K= ',K)
    nData=len(fVal)   #number of observations
    print('...... Number of Data point, n= ',nData)
    #(2) Find the coefficients in the expansion:Only Regression method can be used. 
    #    Also we need to compute gamma_k (=sum2) required to estimate the variance by PCE.
    #    For this we create an auxiliary Gauss-Quadrature grid to compute intgerals
    A=np.zeros((nData,K))    #Matrix of known coeffcient for regression to compute PCE coeffcients
    sum2=[]
    for k in range(K):
        psi_k_=pceBasis(kSet[k][0],xi[0],distType[0])
        sum2_=np.sum(psi_k_**2*w[0])*fac[0]
        aij_=pceBasis(kSet[k][0],xiGrid[:,0],distType[0])
        for i in range(1,p):
            aij_*=pceBasis(kSet[k][i],xiGrid[:,i],distType[i])
            psi_k_=pceBasis(kSet[k][i],xi[i],distType[i])
            sum2_*=np.sum(psi_k_**2*w[i])*fac[i]
        A[:,k]=aij_
        sum2.append(sum2_)
    #Find the PCE coeffs by Linear Regression 
    fCoef=linAlg.myLinearRegress(A,fVal)   #This can be over-, under-, or uniquely- determined systetm.
    #(3) Find the mean and variance of f(q) as estimated by PCE
    fMean=fCoef[0]
    fVar=0.0
    for k in range(1,K):
        fVar+=fCoef[k]*fCoef[k]*sum2[k]   #0.5:PDF of Uniform on [-1,1]
    return fCoef,kSet,fMean,fVar
#
def pce_pd_eval(fk,kSet,xi,distType):
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
    p=len(xi)
    K=len(kSet)
    for k in range(K):        
        a_=fk[k]*pceBasis(kSet[k][0],xi[0],distType[0])
        for i in range(1,p):
            b_=pceBasis(kSet[k][i],xi[i],distType[i])
            a_=np.matmul(np.expand_dims(a_,axis=a_.ndim),b_[None,:])
        if k>0:
           A+=a_ 
        else:
           A=a_ 
    fpce=A    
    return fpce
#    
#
################################
# Tests
################################ 
#
def pce_1d_test():
    """
    Test PCE for 1d uncertain parameter 
    """
    #--- settings -------------------------
    #Parameter settings
    distType='Unif'   #distribution type of the parameter
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
       if distType=='Unif': 
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
    Test PCE for 2 uncertain parameters 
    """
    #---- SETTINGS------------
    distType=['Unif','Unif']   #distribution type of the parameters
    qBound=[[-1,2],   #parameter range 
            [-3,3]] 
    nQ=[11,9]   #number of collocation smaples of param1
    nTest=[121,120]   #number of test points in parameter spaces
    #PCE Options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'     #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten
    if truncMethod=='TO':
       LMax=20   #max polynomial order in each parameter direction
    #------------------------
    p=len(distType)
    #Generate training data
    xi=[]
    q=[]
    for i in range(p):
        xi_,w_=gqPtsWts(nQ[i],distType[i])  #samples from the mapped space
        q_=mapFromUnit(xi_,qBound[i])       #samples from the admissible space
        xi.append(xi_)
        q.append(q_)
    fVal=analyticTestFuncs.fEx2D(q[0],q[1],'type1','tensorProd')  #function value at the parameter samples (Gauss quads)    
    #Construct the gPCE
    pceDict={'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    pceDict=pceDict_corrector(pceDict)
    xiGrid=reshaper.vecs2grid(xi)
    fCoefs,kSet,fMean,fVar=pce_pd_cnstrct(fVal,nQ,xiGrid,pceDict)
    #Plot convergence of PCE terms
    PCE_coef_conv_plot(fCoefs,kSet,distType,{})
    #Make predictions at test points in the parameter space
    qTest=[]
    xiTest=[]
    for i in range(p):
        qTest_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])  
        xiTest_=mapToUnit(qTest_,qBound[i])
        qTest.append(qTest_)
        xiTest.append(xiTest_)
    fTest=analyticTestFuncs.fEx2D(qTest[0],qTest[1],'type1','tensorProd')
    fPCE=pce_pd_eval(fCoefs,kSet,xiTest,distType)  #Prediction at test points by PCE
    #Create 2D grid and response surface over it
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F')
    fErrorGrid=np.zeros((nTest[0],nTest[1]))
    for j in range(nTest[1]):
        for i in range(nTest[0]):
            k=i+j*nTest[0]
            #Compute error between exact and surrogate response
            tmp=fTestGrid[i,j]
            if abs(tmp)<1.e-1:
               tmp=1e-1
            fErrorGrid[i,j]=((abs(fTestGrid[i,j]-fPCE[i,j]))/tmp*100.)
    #2d grid from the sampled parameters
    q1Grid,q2Grid=plot2d.plot2D_grid(q[0],q[1])
    #plot 2d contours
    plt.figure(figsize=(21,8));
    plt.subplot(1,3,1)
    ax=plt.gca()
    CS1 = plt.contour(qTest[0],qTest[1],fTestGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact Response')
    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(qTest[0],qTest[1],fPCE.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Surrogate Response')
    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(qTest[0],qTest[1],fErrorGrid.T,40)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.0f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.plot(q1Grid,q2Grid,'o',color='k',markersize=7)
    plt.title('|Exact-Surrogate|/Exact %')
    plt.show()
#     
def pce_3d_test():
    """
    Test PCE for 3 uncertain parameters
    """
    #----- SETTINGS------------
    distType=['Unif','Unif','Unif']
    qBound=[[-0.75,1.5],   #range of param1
             [-0.5,2.5],   #range of param2
             [ 1.0,3.0]]   #range of param3
    nQ=[6,5,4] #number of parameter samples in the 3 dimensions
    funOpt={'a':7,'b':0.1}   #parameters in Ishigami function
    #PCE options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'   #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'Regression' can be selected
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
    for i in range(p):
        xi_,w_=gqPtsWts(nQ[i],distType[i])  #samples from the mapped space
        q_=mapFromUnit(xi_,qBound[i])       #samples from the admissible space
        xi.append(xi_)
        q.append(q_)
    fVal=analyticTestFuncs.fEx3D(q[0],q[1],q[2],'Ishigami','tensorProd',funOpt)  
    #construct the gPCE and compute moments
    pceDict={'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    pceDict=pceDict_corrector(pceDict)
    xiGrid=reshaper.vecs2grid(xi)
    fCoefs,kSet,fMean,fVar=pce_pd_cnstrct(fVal,nQ,xiGrid,pceDict)
    #plot convergence of PCE terms
    PCE_coef_conv_plot(fCoefs,kSet,distType,{})
    #exact moments of Ishigami function
    m,v=analyticTestFuncs.ishigami_exactMoments(qBound[0],qBound[1],qBound[2],funOpt)
    #print the results
    print(writeUQ.printRepeated('-',50))
    print('\t\t Exact \t\t PCE')
    print('E[f]:  ',m,fMean)
    print('V[f]:  ',v,fVar)
    #Compare the PCE predictions with exact values of the model response
    qTest=[]
    xiTest=[]
    for i in range(p):
        qTest_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])  
        xiTest_=mapToUnit(qTest_,qBound[i])
        qTest.append(qTest_)
        xiTest.append(xiTest_)
    fVal_test_ex=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',funOpt)  
    fVal_test_pce=pce_pd_eval(fCoefs,kSet,xiTest,distType)
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

