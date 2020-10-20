####################################################################
# Read in  and construct databases from 
#      the statistics profiles of turbulent channel flow
####################################################################
#   Saleh Rezaeiravesh, salehr@kth.se
#-------------------------------------------------------------------
import sys
import os
import numpy as np
import math as mt
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from lagInt import lagInt
#
def lagrangeInterpGLL2UnifMesh(y,f,interpOpts):
    """
    Lagrange interpolation from GLL points at each NEK5000 element to an arbitrary number of uniformly-spaced points inside the same element. 
    f(y) is mapped into fInt(yInt)

    Args:
      `y`: 1D numpy array 
         Containing the global coordinates of the GLL points
      `f`: 1D numpy array
         Values at `y`
      `interpOpts`: dict   
         Options for interpolation
    
    Returns:
      `yInt`: 1D numpy array
         Containing the global coordinates of yInt
      `fInt`: 1D numpy array
         Interpolated f at `yInt`
    """
    nGLL=interpOpts['nGLL']   
    nTot=len(y)                 
    nElement=int(nTot/nGLL)    
    nIntPerE=interpOpts['nIntPerE'] 

    yGLL=np.zeros(nGLL)   
    fGLL=np.zeros(nGLL)
    yInt=[]
    fInt=[]
    for i in range(nElement):
        for j in range(nGLL):
            k=i*nGLL+j 
            yGLL[j]=y[k]               
            fGLL[j]=f[k]               
        yInt_E=np.linspace(np.min(yGLL),np.max(yGLL),nIntPerE)
        fInt_E=lagInt(fNodes=fGLL,qNodes=[yGLL],qTest=[yInt_E]).val
        for j in range(nIntPerE):
            yInt.append(yInt_E[j])
            fInt.append(fInt_E[j])
    yInt=np.asarray(yInt)
    fInt=np.asarray(fInt)
    return yInt,fInt
#
def sortedCaseInfo(caseDir,caseName):
    """ 
    Read information about a Nek5000 case (a set of channel flow simulations)
    The info is saved in a database which will be sorted to make sure our convention for tensor-product in parameter space holds. This sorted info-db will be later used to sort other databases of the case.
    The info file contains information about uncertain parameters associated with each case and has the following format.

      # Information on the simlations set <caseName>. \n
      # caseName: \n
      # Number of included simulations: \n
      # Number of uncertain parameters: \n
      # Number of parameter samples in each direction:  (non-tensor-product: nTotSamples 1 1 ... 1) \n
      # Name of uncertain parameters: \n
      # Admissible range of parameters: [q1Min q1Max] [q2Min q2Max] \n
      #List of simulations name and associated parameter values \n
      # simName    q1      q2 \n
      #--------------------------------------------------------- (iEnd1)
    
    Args:
      `caseDir`: string
         Path to the Nek5000 simulations data
      `caseName`: string
         Name of the simulation case
    
    Returns:
      `db_info`: dict
         Information about the Nek5000 set of simulations
    """   
    caseName=caseName+'_info.dat'
    F1=open(caseDir+caseName,'r')     
    ain=F1.readlines();
    ain_sep=[];
    for i in range(len(ain)):
        ain_sep.append(ain[i].split())
    #(1) Read the info in the header
    nSim=int(ain_sep[2][-1])   
    nPar=int(ain_sep[3][-1])   
    nSamples=[]               
    parName=[]
    parRange=[]              
    parVals=[]
  
    for i in range(nPar):
        nSamples.append(int(ain_sep[4][-nPar+i]))   
        parName.append(ain_sep[5][-nPar+i])   
        rNo_=6   
        r1_=float(ain_sep[rNo_][4+(2*i+1)][1:])  
        r2_=float(ain_sep[rNo_][4+(2*i+2)][:-1])
        parRange.append([r1_,r2_])
        parVals.append([])

    iEnd1=10   
    #(2) Read the info in the list
    simName=[]
    for i in range(nSim):
        I=iEnd1+i
        simName.append(ain_sep[I][0])
        for j in range(nPar):
            parVals[j].append(float(ain_sep[I][-nPar+j]))
    #(3) Make a database from the read info
    db_info={'simName':simName,
             'nSims':nSim,
             'parName':parName,
             'parRange':parRange,
             'nSamples':nSamples}            
    for i in range(nPar):
        db_info.update({('q'+str(i+1)):parVals[i]})
    #(4) Sort the db_info
    if nPar>1:
       quads=quads_tensorProd_params(parVals)
       tmp={x:[] for x in db_info.keys()}
       if nPar==2:
          for Q2 in (quads[1]):
              for Q1 in (quads[0]):
                  for i in range(nSim):          
                      q1=parVals[0][i]
                      q2=parVals[1][i]
                      if q1==Q1 and q2==Q2:
                         tmp['simName'].append(db_info['simName'][i])
                         for j in range(nPar):
                             tmp['q'+str(j+1)].append(db_info['q'+str(j+1)][i])
                         break
       elif nPar==3:
            for Q3 in (quads[2]):
                for Q2 in (quads[1]):
                    for Q1 in (quads[0]):
                        for i in range(nSim):          
                            q1=parVals[0][i]
                            q2=parVals[1][i]
                            q3=parVals[2][i]
                            if q1==Q1 and q2==Q2 and q3==Q3:
                               tmp['simName'].append(db_info['simName'][i])
                               for j in range(nPar):
                                   tmp['q'+str(j+1)].append(db_info['q'+str(j+1)][i])
                               break
       else:
         raise ValueError('Currently nPar<4 is supported. This can be modified.')
       tmp['parName']=db_info['parName']               
       tmp['nSims']  =db_info['nSims']
       tmp['parRange']=db_info['parRange']               
       tmp['nSamples']=db_info['nSamples']               
       db_info=tmp
    return db_info
#
def quads_tensorProd_params(qList):
    """
    Extract parameter samples (not necessarily Gauss-quadrature nodes) in each direction in the parameter space given `qList`

    Args:
       `qList`: List of length p (number of parameters)
          =[[q1],[q2],...,[qp]] where qi are the parameter samples for the i-th parameter.
    
    Returns:
       `quads`: List of length p
          Containing sorted non-repeated parameter samples in each direction in parameter space
    """
    def nonRepeatedEl(x):
        """
        Find non-repeated elements of x
        """
        nx=len(x)
        xUniq=[x[0]]
        for i in range(nx):
            ifac=0
            for j in range(len(xUniq)):
                if x[i]==xUniq[j]:
                   ifac=1
            if ifac==0:
               xUniq.append(x[i])    
        return xUniq
    
    nPar =len(qList)
    nSamp=len(qList[0])
    quads=[]   
    for i in range(nPar):
        quads.append(sorted(nonRepeatedEl(qList[i])))
    return quads
#
def sort_merge_dbs(dbSim,db_info):
    """
    Sorts `dbSim` according to `db_info['simName']` and then merges `dbSims` and `db_info`. 
    The sorting is done to ensure that our convention for the tensor product of parameter samples
    in spaces of dimensionality of >= 2 holds (order='F', Fortran-like loops over parameters). 

    Args:
      `dbSims`: List of size N
         Each element is a dict that contains the QoIs of a channel flow simulation. 
         This list of databases represents a simulation case (in computer experiment). 
      `db_info`: dict
         Containing information about the simualtion case including the parameter samples corresponding to which channel flow simulations have been performed. 
         This dict is already sorted to ensure the convention of Fortran-like loops (order='F'). 
    
    Returns:
      `db`: List 
         Merged `dbSim` and `db_info`
    """
    nSim=len(dbSim)               
    nPar=len(db_info['parName'])  
    db=[]
    #sorting & merging
    for j in range(nSim):               
        for i in range(nSim):
            if (db_info['simName'][j]==dbSim[i]['name']):
               tmp_=dbSim[i]
               tmp_.update({'parName':db_info['parName']})  
               parVal=[]   
               parValMap=[] 
               for k in range(nPar):
                   parVal_=db_info['q'+str(k+1)][j]
                   parRange1_=db_info['parRange'][k][0]
                   parRange2_=db_info['parRange'][k][1]
                   xi_=-1+2.*(parVal_-parRange1_)/(parRange2_-parRange1_)
                   parVal.append(parVal_)
                   parValMap.append(xi_)    
               tmp_.update({'parVal':parVal})  
               tmp_.update({'parValMapped':parValMap})  
               db.append(tmp_)
               break
    return db
#    
def dbCnstrctr_NEK(dirNekData,interpOpts):
    """ 
    Construct database (dicts) from the QoIs of a single Nek5000 channel flow simulation. 

    Args:
       `dirNekData`: string
          Path to the Nek5000 data of the channel flow QoIs
       `interpOpts`: dict
          Options for interpolating original Nek5000 profiles to a uniform mesh.
    Returns: 
       `dataNek`: dict
          Dictionary of the channel flow QoIs.          
    """
    [uTau,y,yp,up]=dataReader_1stOstats(dirNekData)
    u=up*uTau
    [y,yp,uup,vvp,wwp,uvp]=nekDataReader_2ndOstats(dirNekData)
    uu=uup*uTau
    vv=vvp*uTau
    ww=wwp*uTau
    vv=vvp*uTau
    uv=uvp*uTau**2.
    tke=0.5*(uu**2.+vv**2.+ww**2.)
    tkep=tke/uTau**2.
    if interpOpts['doInterp']: 
       print('... Nek5000 profiles are interpolated to uniform fine mesh.')       
       yInt,yp =lagrangeInterpGLL2UnifMesh(y,yp,interpOpts)       
       yInt,u  =lagrangeInterpGLL2UnifMesh(y,u,interpOpts)
       yInt,up =lagrangeInterpGLL2UnifMesh(y,up,interpOpts)
       yInt,uu =lagrangeInterpGLL2UnifMesh(y,uu,interpOpts)
       yInt,uup=lagrangeInterpGLL2UnifMesh(y,uup,interpOpts)
       yInt,vv =lagrangeInterpGLL2UnifMesh(y,vv,interpOpts)
       yInt,vvp=lagrangeInterpGLL2UnifMesh(y,vvp,interpOpts)
       yInt,ww =lagrangeInterpGLL2UnifMesh(y,ww,interpOpts)
       yInt,wwp=lagrangeInterpGLL2UnifMesh(y,wwp,interpOpts)
       yInt,uv =lagrangeInterpGLL2UnifMesh(y,uv,interpOpts)
       yInt,uvp=lagrangeInterpGLL2UnifMesh(y,uvp,interpOpts)
       yInt,tke=lagrangeInterpGLL2UnifMesh(y,tke,interpOpts)
       yInt,tkep=lagrangeInterpGLL2UnifMesh(y,tkep,interpOpts)
       y=yInt
    dataNEK={"uTau":uTau,"y":y,"y+":yp,"u":u,"u+":up,\
             "u'":uu,"u'+":uup,"v'":vv,"v'+":vvp,"w'":ww,"w'+":wwp,\
             "tke":tke,"tke+":tkep,"uv":uv,"uv+":uvp}
    return dataNEK
#
def dataReader_1stOstats(dirNekData):
    """
    Read in uTau, y, y+, and <u>+ of turbulent channel flow simulated by Nek5000.

    Args:
      `dirNekData`: string
         Path to  mean_prof.dat
    """
    F1=open(dirNekData+'/'+'mean_prof.dat','r')

    ain=F1.readlines();
    ain_sep=[];
    for i in range(len(ain)):
        ain_sep.append(ain[i].split())

    iskip=4;  
    n=len(ain_sep)-iskip;  
    yNek=np.zeros(n);
    ypNek=np.zeros(n);
    upNek=np.zeros(n);
    for i in range(n):
        yNek[i]=float(ain_sep[iskip+i][0]);
        ypNek[i]=float(ain_sep[iskip+i][1]);
        upNek[i]=float(ain_sep[iskip+i][2]);
    uTauNek=float(ain_sep[1][2])
    return uTauNek,yNek,ypNek,upNek
#
def nekDataReader_2ndOstats(dirNekData):
    """
    Read in velocity 2nd-order statistical moments of turbulent channel flow simulated by Nek5000.

    Args:
      `dirNekData`: string
         Path to vel_fluc_prof.dat
    """
    F1=open(dirNekData+'/'+'vel_fluc_prof.dat','r')

    ain=F1.readlines();
    ain_sep=[];
    for i in range(len(ain)):
        ain_sep.append(ain[i].split())

    iskip=4;  
    n=len(ain_sep)-iskip;  
    yNek=np.zeros(n);
    ypNek=np.zeros(n);
    uupNek=np.zeros(n);
    vvpNek=np.zeros(n);
    wwpNek=np.zeros(n);
    uvpNek=np.zeros(n);
    for i in range(n):
        yNek[i]=float(ain_sep[iskip+i][0]);
        ypNek[i]=float(ain_sep[iskip+i][1]);
        uupNek[i]=float(ain_sep[iskip+i][2]);
        uupNek[i]=mt.sqrt(uupNek[i]);   
        vvpNek[i]=float(ain_sep[iskip+i][3]);
        vvpNek[i]=mt.sqrt(vvpNek[i]);   
        wwpNek[i]=float(ain_sep[iskip+i][4]);
        wwpNek[i]=mt.sqrt(wwpNek[i]);   
        uvpNek[i]=-float(ain_sep[iskip+i][5]);
    return yNek,ypNek,uupNek,vvpNek,wwpNek,uvpNek
#
