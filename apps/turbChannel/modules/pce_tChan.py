####################################################################
#   gPCE for simulation of turbulent channel flow
####################################################################
# Saleh Rezaeiravesh, salehr@kth.se
#-------------------------------------------------------------------
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
    Construct a PCE of a QoI (scalar or wall-normal profile) given a database of channel flow simulations. Each simulation correpsponds a sample in the parameter space. 
    NOTE: R=f(q,chi), q:uncertain param, chi: controlled parameter, R: response 
    NOTE: if len(nQ)>1: multi-dimensional parameter space => handled by tensor product
    NOTE: Now ONLY for Uniformly distributed uncertain parameter 
       input:
          db: a list contaning the data of CFD simulations
          qoiName: name of the QoI whose PCE is to be constructed
          nQ: list of parameter samples; 
              if 'TP': nQ=[nQ1,nQ2,...,nQp], nQi=number of samples in the i-th direction
              if 'TO': nQ=[nQtot,1,1,...,1], nQtot: total number of samples
          pceDict: A dictionary containing different options for PCE in multi-dimensional parameter space with these keys:   
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
       output:
          pceCoefs: Coefficients of the PCE
          pceMean,pceVar: mean and variance of the surrogate predicted by PCE
    """
    def xiGridMaker(nSets,nDim,db):
        """
           Construct a grid of mapped samples which are defined in [-1,1]^nDim
        """
        xiGrid=np.zeros((nSets,nDim))
        for i in range(nSets):
            for j in range(nDim):
                xiGrid[i,j]=db[i]['parValMapped'][j]
        return xiGrid

    #------ Decode info and assignments
    nSets=len(db)  #number of sets of data
    nDim=len(nQ)   #dimensionality of the parameter space
    varCheck=db[0][qoiName]   #variable to check
    if not isinstance(varCheck,(list,tuple,np.ndarray)):  # if true, the QoI is scalar
       nChi=1 #scalar QoI
    else:
       nChi=len(varCheck)   #length of QoI profile = length of the controlled parameter vector
                            #NOTE: all cases should have the same nChi, if not do interpolation to a fixed vector

    #------ Construc the gPCE
    pceCoefs=[]  #list of outputs, since the qoi can be a profiles of length (nChi)
    pceMean=[]
    pceVar=[]
    pceCI=[]
    kSet=[]
    for j in range(nChi):
        fVal=[]   #response value at samples        
#        qVal=[]   #parameter samples
        for i in range(nSets):
            #qVal.append(db[i]['q1'])
            if nChi>1:  #profile QoI
                 ##fVal.append(db[i][qoiName][j][0])
                 fVal.append(db[i][qoiName][j])
            else:     #scalar QoI
                 fVal.append(db[i][qoiName])
        fVal=np.asarray(fVal)         
        if nDim==1:  #1d parameter space
           xi=np.zeros(0)
           for i in range(nSets):
                xi=np.append(xi,[db[i]['parValMapped'][0]]) 
           xi=np.asarray(xi)     
           pce_=pce.pce(fVal=fVal,xi=xi[:,None],pceDict=pceDict)
           kSet=[]  #dummy
        elif nDim>1:
           xiGrid=xiGridMaker(nSets,nDim,db)
           pce_=pce.pce(fVal=fVal,xi=xiGrid,nQList=nQ,pceDict=pceDict)
           kSet_=pce_.kSet
        pceMean_=pce_.fMean  #mean, var estimated by the PCE and PCE coefficients
        pceVar_=pce_.fVar
        pceCoefs_=pce_.coefs          

        pceCoefs.append((pceCoefs_))
        pceMean.append(pceMean_)
        pceVar.append(pceVar_)
        if nDim>1:
           kSet.append(kSet_)
        pceCI.append(1.96*mt.sqrt(pceVar_))
    pceMean=np.asarray(pceMean)
    pceVar=np.asarray(pceVar)
    pceCI=np.asarray(pceCI)
    return pceCoefs,kSet,pceMean,pceVar,pceCI
#
