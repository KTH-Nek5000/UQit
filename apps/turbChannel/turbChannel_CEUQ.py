#***************************************************************************
# Application of UQit to computer experiments of turbulent channel flow
# The uncertain parameters are grid resolutions (scaled in wall-units) in 
#   the wall-parallel directions
#***************************************************************************
#
import os
import sys
import numpy as np
sys.path.append(os.getenv("UQit"))
import pce
import sobol
sys.path.append("./modules/")
import dbMaker_tChan
import pce_tChan
import plot_tChan
import nekProfsReader_tChan
from texOptions_tChan import texLabel
#
# CFD data
dataPath="./data/C8/"          #Link to the channel flow data
figDir="../../testFigs/apps/"  #Link to save the plots
caseName='C8'
#
def CI_sobol_ChanProfs():
    """
    Construct confidence intervals and compute Sobol indices for the channel flow 
    QoIs for 2 uncertain parameters.
    """
    #----SETTINGS------------------------
    qoiName="u+"  #name of the channel QoI
    chiName="y+"   #name of the controlled parameter (wall-normal coordinate)
    #>>> PCE Options
    truncMethod='TP'  # truncation scheme, 'TP'=Tensor Product, 'TO'=Total Order
    sampleType='GQ'   #type of parameter samples: 'GQ'=Gauss Quadrature nodes
    pceSolveMethod='Regression' #method to solve for the PCE coefficients
                                # 'Projection': 'GP'+'TP'
                                # 'Regression': any combination 
    if truncMethod=='TO':
       LMax=12   #max polynomial order in each parameter direction
    #>>> Settings for Sobol indices   
    qName=[r'$\Delta x^+$',r'$\Delta z^+$']  #parameter labels
    nq_test=[30,30]    #number of test points to compute Sobol indices
    #>>> plot options
    xLim=[0.5,300]   #Limits of the horizontal axis (optional)
    legend_sobol='on' 
    #yLim=[0,25]
    #------------------------------------
    #
    # settings for interpolating from Nek5000 GLL points to a uniform mesh
    interpOpts={'doInterp':True,  
                'nGLL':8,         #number of GLL points per element in the Nek5000 simulations
                'nIntPerE':30     #number of uniformly-spaced points per Nek5000 element
               }
    # (1) Read in channel flow data included in the computer experiment
    db,nSamples=dbMaker_tChan.dbMakerCase_multiPar(dataPath,caseName,interpOpts)
    p=len(db[0]['parName']) 
    distType=['Unif']*p
    #(2) Construct gPCE for different QoI 
    #  (a) Assign the pceDict
    pceDict={'p':p,'truncMethod':truncMethod,'sampleType':sampleType,
            'pceSolveMethod':pceSolveMethod,'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    #  (b) PCE for uTau
    pceCoefs,kSet,pceMean,pceVar,pceCI=pce_tChan.pce_chan(db,'uTau',nSamples,pceDict)
    pce.convPlot(coefs=pceCoefs[0],distType=distType,kSet=kSet[0])
    print('E(uTau)=%g, CI(uTau)=[%g,%g]' %(pceMean,pceMean-pceCI,pceMean+pceCI))
    #  (c) PCE for a vector QoI (profile) specified by qoiName
    pceCoefs,kSet,pceMean,pceVar,pceCI=pce_tChan.pce_chan(db,qoiName,nSamples,pceDict)
    #  (d) plot convergence of the PCE terms at point ichi of the profile of the QoI
    iChi=30 
    pce.convPlot(coefs=pceCoefs[iChi],distType=distType,kSet=kSet[iChi])
    #(3) Plot and save the figure
    #   (a) set the plot options
    pltOpts={'qoiLabel':texLabel(qoiName),
             'chiLabel':texLabel(chiName),
             'xAxisScale':'log', #'log' or ''
             'figName':caseName+'_'+qoiName,
             'figDir':figDir,'legend':'on'}
    #   (b) update the plot xlim and ylim
    try:
       xLim
    except NameError:
       print('Just to know: No xlim is set for plotting =)')
    else:
       pltOpts.update({'xLim':xLim})
    #
    try:
       yLim
    except NameError:
       print('Just to know: No ylim is set for plotting =)')
    else:
       pltOpts.update({'yLim':yLim})
    #
    #  (c) set the values at the plot horizontal axis
    if chiName=='y':   #horizontal axis=y
       hrzAxVals=db[0][chiName]   #assume all cases have the same-size y-vectors
    if chiName=='y+':  #horizontal axis =E[y+]
       pceCoefs_,kSet_,pceMean_,pceVar_,pceCI_=pce_tChan.pce_chan(db,chiName,nSamples,pceDict)
       hrzAxVals=pceMean_
    pltOpts.update({'hrzAxVals':hrzAxVals})
    #(4) Compute and plot CI on the profile of qoiName vs. chiName
    plot_tChan.plotUQChan_profCI(db,chiName,qoiName,pceMean,pceCI,pltOpts)
    #(5) Sobol indices
    # (a) settings
    db_info=nekProfsReader_tChan.sortedCaseInfo(dataPath,caseName) 
    qBound=db_info['parRange']   
    nSims=db_info['nSims']
    nChi=len(pceCoefs)
    pdf=[] #parameters' PDF
    for i in range(p):
        N_=nq_test[i]
        qBound_=db_info['parRange'][i]
        pdf.append(np.ones(N_)/(qBound_[1]-qBound_[0]))
    if db_info['nSamples']:   
       nQ_pce=db_info['nSamples'] 
    else:    
       nQ_pce=[nSims]   
       for i in range(1,p):
           nQ_pce.append[1]
    #total number of test samples    
    nq_test_tot=1
    for i in range(p):        
        nq_test_tot*=nq_test[i]
    # (b) test samples in the parameter space
    qTest=[]
    xiTest=[]
    for i in range(p):
        qTest.append(np.linspace(qBound[i][0],qBound[i][1],nq_test[i]))
        xiTest.append(pce.pce.mapToUnit(qTest[i],qBound[i]))
    # (c) compute and plot Sobol indices
    S1=[]    #Main Sobol indices
    S2=[]
    S12=[]   #2nd-order interaction
    print('... Computing Sobol sensitivity indices for %s wrt %d uncertain parameters.' %(qoiName,p))
    for I in range(1,nChi):   #points on the profile of qoi (ignore y=0 since returns nan)
        pcePred_=pce.pceEval(coefs=pceCoefs[I],xi=xiTest,distType=distType,kSet=kSet[I])
        fTest=pcePred_.pceVal
        sobol_=sobol.sobol(qTest,fTest,pdf)
        S1.append(sobol_.Si[0])
        S2.append(sobol_.Si[1])
        S12.append(sobol_.Sij[0])
    chi=db[0][chiName][1:]  
    figOpts={'figDir':figDir,
             'figName': 'sobol_'+caseName+'_'+qoiName+'.pdf',
             'parLabs':qName,
             'xLogScale':True,
             'figSize':[800,400],
             'xTicks_fs':17,
             'yTicks_fs':17,
             'xLim':xLim,
             'legend':legend_sobol
            }
    plot_tChan.plotUQChan_profSobol(chi,S1,S2,[],S12,[],[],p,chiName,figOpts)    
#
