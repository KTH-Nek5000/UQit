#################################################
#   gPCE for UQ of turbulent channel flow
#################################################
# Saleh Rezaeiravesh, salehr@kth.se
#------------------------------------------------
#
import sys
import os
import numpy as np
import math as mt
sys.path.append(os.getenv("UQit"))
import pce
#
def pce_chan(db,qoiName,nQ,pceDict):
    """ 
    Construct a gPCE for a QoI (scalar or wall-normal profile) given a database of channel flow simulations. 

    Args:
      `db`: List
         Contaning the data of the channel flow simulations in the computer experiment
      `qoiName`: string
         Name of the QoI whose PCE is to be constructed
      `nQ`: List of length p
         If 'truncMethod'=='TP', then nQ=[nQ1,nQ2,...,nQp], nQi=number of samples in the i-th direction
         If 'truncMethod'=='TO',then nQ=[nQtot,1,1,...,1], nQtot: total number of samples
      `pceDict`: dict
         Contains settings required for constructing the PCE. The keys are:
            * `'p'`: int
               Dimension of the parameter
            * `'distType'`: List of length p, 
               The i-th value specifies the distribution type of the i-th parameter (based on the gPCE rule)
            * `'sampleType'`: string 
               Type of parameter samples at which observations are made
                - `'GQ'` (Gauss quadrature nodes)
                - `' '`  (other nodal sets, see `class trainSample` in `sampling.py`)
            * `'pceSolveMethod'`: string
               Method of solving for the PCE coefficients
                - `'Projection'`: Projection method; samples have to be Gauss-quadrature nodes.
                - `'Regression'`: Regression method for uniquely-, over-, and under-determined systems.
                   If under-determined, compressed sensing with L1/L2 regularization is automatically used.
            * `'truncMethod'`: string (optional, only for p>1) 
                Method of truncating the PCE
                 - `'TP'`: Tensor-Product method     
                 - `'TO'`: Total-Order method
            * `'LMax'`: int (optional)
                Maximum order of the PCE in each of the parameter dimensions. 
                 It is mandatory for p>1 and `'TuncMethod'=='TO'`
                  - `'LMax'` can be used only with `'pceSolveMethod':'Regression'`
                  - If p==1 and `'LMax'` is not provided, it will be assumed to be equal to n.
                  - If p>1 and `'LMax'` is not provided, it will be assumed to a default value.   

       Returns:
          `pceCoefs`: List of length nChi (number of points in the wall-normal direction)
             Each member is a 1D numpy array of size K containing the PCE coefficients
          `pceMean`: 1D numpy array of size nChi
             Each memeber is a scalar representing the mean of the QoI estimated by PCE
          `pceVar`: 1D numpy array of size nChi
             Each memeber is a scalar representing the variance of the QoI estimated by PCE
          `pceCI`: 1D numpy array of size nChi
             Each memeber is a scalar representing the 95% CI of the QoI estimated by PCE
          `kSet`: List of length nChi
             Each member is a list of length K containing the PCE index set
    """
    def xiGridMaker(nSets,nDim,db):
        """
        Construct a grid of mapped samples which defined in [-1,1]^p
        """
        xiGrid=np.zeros((nSets,nDim))
        for i in range(nSets):
            for j in range(nDim):
                xiGrid[i,j]=db[i]['parValMapped'][j]
        return xiGrid
    #
    nSets=len(db)  #number of data sets
    nDim=len(nQ)   #number of parameters
    varCheck=db[0][qoiName]   
    if not isinstance(varCheck,(list,tuple,np.ndarray)):  
       nChi=1 
    else:
       nChi=len(varCheck)   
    #   
    pceCoefs=[]  
    pceMean=[]
    pceVar=[]
    pceCI=[]
    kSet=[]
    for j in range(nChi):
        fVal=[]   
        for i in range(nSets):
            if nChi>1:  
                 fVal.append(db[i][qoiName][j])
            else:     
                 fVal.append(db[i][qoiName])
        fVal=np.asarray(fVal)         
        if nDim==1:  
           xi=np.zeros(0)
           for i in range(nSets):
                xi=np.append(xi,[db[i]['parValMapped'][0]]) 
           xi=np.asarray(xi)     
           pce_=pce.pce(fVal=fVal,xi=xi[:,None],pceDict=pceDict)
           kSet=[]  
        elif nDim>1:
           xiGrid=xiGridMaker(nSets,nDim,db)
           pce_=pce.pce(fVal=fVal,xi=xiGrid,nQList=nQ,pceDict=pceDict)
           kSet_=pce_.kSet
        pceMean_=pce_.fMean 
        pceVar_=pce_.fVar
        pceCoefs_=pce_.coefs          

        pceCoefs.append((pce_.coefs))
        pceMean.append(pce_.fMean)
        pceVar.append(pce_.fVar)
        if nDim>1:
           kSet.append(kSet_)
        pceCI.append(1.96*mt.sqrt(pceVar_))
    pceMean=np.asarray(pceMean)
    pceVar=np.asarray(pceVar)
    pceCI=np.asarray(pceCI)
    return pceCoefs,kSet,pceMean,pceVar,pceCI
#
