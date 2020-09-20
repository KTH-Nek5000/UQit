#*********************************************************************
# Application of UQit to computer experiments of turbulent channel flow
# The uncertain parameters are grid resolutions in wall-parallel directions
#     described in the wall-unit
#*********************************************************************
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
dataPath="./data/C8/"    #link to the data
figDir="../../testFigs/apps/"  #link to save the plots
caseName='C8'
#
def CI_sobol_ChanProfs():
    """
    Construct confidence intervals for the channel flow QoIs for 2 uncertain parameters.
    """
    #----SETTINGS------------------------
    qoiName="u'+"   #name of the QoI
                   #NOTE2: to get the list of qoi: print(db[0].keys())
    chiName="y+"   #name of the controlled parameter
    #>>> PCE Options
    truncMethod='TP'  #'TP'=Tensor Product
                      #'TO'=Total Order
    sampleType='GQ'     #'GQ'=Gauss Quadrature nodes
                      #''= any other sample => only 'Regression' can be selected
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for GQ+Tensor Product
    #NOTE: for 'TO' only 'Regression can be used'. pceSolveMethod will be overwritten, anyway.
    if truncMethod=='TO':
       LMax=12   #max polynomial order in each parameter direction

    #Settings for Sobol indices   
    qName=[r'$\Delta x^+$',r'$\Delta z^+$']  #for Sobol indices plot labels
    nq_test=[30,30]

    #>>> plot options
    xLim=[0.5,300]   #you can comment these, if you do not need them
    legend_sobol='on'

#    yLim=[0,25]
    #------------------------------------
    #
    #>>> Nek interpolation options
    interpOpts={'doInterp':True,  #Interpolating original Nek profiles on a mesh that is uniform on each Nek element. In each elements GLL points are used in Lagrange interpolation.
                'nGLL':8,        #number of GLL points per element in the Nek simulations
                'nIntPerE':30     #number of interpolation points per Nek element
               }
    #(1) Read in the list of channel flow post-processed databases. Each simulation has one database (dict).
    db,nSamples=dbMaker_tChan.dbMakerCase_multiPar(dataPath,caseName,interpOpts)
    p=len(db[0]['parName'])   #number of parameters
    distType=['Unif']*p
    #(2) Construct PCE for different QoI using the Gauss samples for (param1, param2)
    #  (a) Assign the pceDict
    pceDict={'p':p,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    #  (b) PCE for scalar QoI
    pceCoefs,kSet,pceMean,pceVar,pceCI=pce_tChan.pce_chan(db,'uTau',nSamples,pceDict)
    pce.convPlot(coefs=pceCoefs[0],distType=distType,kSet=kSet[0])
    print('E(uTau)=%g, CI(uTau)=[%g,%g]' %(pceMean,pceMean-pceCI,pceMean+pceCI))
    #  (c) PCE for vector QoI (profiles) whose name is specified by qoiName
    pceCoefs,kSet,pceMean,pceVar,pceCI=pce_tChan.pce_chan(db,qoiName,nSamples,pceDict)
    #  (d) plot convergence of PCE coeffcients at iChi point in the profile of the QoI
    iChi=30   #point in the profile for which convergence of PCE coeffcients must be plotted
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
    #   (4) Compute and plot CI on the profile of qoiName vs. chiName
    plot_tChan.plotUQChan_profCI(db,chiName,qoiName,pceMean,pceCI,pltOpts)


    #Sobol indices
    db_info=nekProfsReader_tChan.sortedCaseInfo(dataPath,caseName) #info about the simulatons set
    qBound=db_info['parRange']   #admissible range of parameters, assume range is the same for all simulations in the db
    nSims=db_info['nSims']
    nChi=len(pceCoefs)
    #Make PDFs for the parameters (for Sobol indices)
    pdf=[]
    for i in range(p):
        N_=nq_test[i]
        qBound_=db_info['parRange'][i]
        pdf.append(np.ones(N_)/(qBound_[1]-qBound_[0]))

    if db_info['nSamples']:   #tensor-product grid
       nQ_pce=db_info['nSamples'] #list of number of samples in each direction
    else:    #non-tensor-product parameter samples 
       nQ_pce=[nSims]   
       for i in range(1,p):
           nQ_pce.append[1]
    #total number of test samples    
    nq_test_tot=1
    for i in range(p):        
        nq_test_tot*=nq_test[i]

    #(3) use gPCE to predict at test samples in parameter space
    qTest=[]
    xiTest=[]
    for i in range(p):
        qTest.append(np.linspace(qBound[i][0],qBound[i][1],nq_test[i]))
        xiTest.append(pce.pce.mapToUnit(qTest[i],qBound[i]))
  #List of Sobol indices
    S1=[]    #Main Sobol indices
    S2=[]
    S12=[]   #2nd-order interaction
    print('... Computing Sobol sensitivity indices for %s wrt %d uncertain parameters.' %(qoiName,p))
    for I in range(1,nChi):   #points on the profile of qoi (ignore y=0 since returns nan)
        if (I%10==0):
           print('...... %d%% done!' %(float(I)/nChi*100))
        #(i) Evaluate PCE at test points in the parameter space
        pcePred_=pce.pceEval(coefs=pceCoefs[I],xi=xiTest,distType=distType,kSet=kSet[I])
        fTest=pcePred_.pceVal
        #Plot convergence of PCE coeffcients at iChi point in the profile of the QoI
        iChi=10  #point in the profile at which convergence of PCE is investigated 
        if I==iChi:
           pce.convPlot(coefs=pceCoefs[I],distType=distType,kSet=kSet[I])

        #Compute Sobol indices 
        sobol_=sobol.sobol(qTest,fTest,pdf)
        S1.append(sobol_.Si[0])
        S2.append(sobol_.Si[1])
        S12.append(sobol_.Sij[0])

    chi=db[0][chiName][1:]  #count from 1, since we skipped the wall in the above loop (since Sobol indices would become =nan)
    pceMean_qoi=pceMean[1:]  #count from 1, since we skipped the wall in the above loop (since Sobol indices would become =nan)
    #(4i) Plot contours of PDF of qoi in the qoi-chi plane. 
    xLab_=texLabel(chiName)
    if chiName=="y+":
       xLab_=r'$\mathbb{E}_{\mathbf{q}}[$'+xLab_+'$]$' 
    # (5) Plot the Sobol indices
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
