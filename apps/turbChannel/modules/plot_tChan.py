####################################################################
# Plotters for UQ of turbulent Channel Flow
####################################################################
#  Saleh Rezaeiravesh, salehr@kth.se
#-------------------------------------------------------------------
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
import reshaper
from lagInt import lagInt
import pce
import texOptions_tChan
#
def pltOpts_default(pltOpts):
    """
    Checks `pltOpts` for essential keys. Default values are assigned for missing values. 
    No overwritting of the existing values in `pltOpts`.
    """
    if 'figSize' not in pltOpts.keys():
       pltOpts['figSize']=[800,400]
    if 'x_fs' not in pltOpts.keys():   #x-label fontsize
       pltOpts['x_fs']=22    
    if 'y_fs' not in pltOpts.keys():   #y-label fontsize
       pltOpts['y_fs']=22    
    if 'xTicks_fs' not in pltOpts.keys():   #x-ticks fontsize
       pltOpts['xTicks_fs']=20
    if 'yTicks_fs' not in pltOpts.keys():   #y-ticks fontsize
       pltOpts['yTicks_fs']=20
#
def plotUQChan_profCI(db,chiName,qoiName,pceMean,pceCI,pltOpts):
    """
    Plot a profile of channel flow QoI along with associated 95%CI due to uncertain parameters. 
        db: list, database
        chiName: controlled variable = Quantity on the horizontal axis, e.g. channel wall-normal coordinate
        qoiName: name of the QoI whose profile is plotted
        pceMean: mean of the QoI predicted by gPCE
        pciCI: 95% confidence interval
        pltOpts: dictionary, plot options     
    """
    #set the values at the horizontal axis
    y=pltOpts['hrzAxVals']
    ist=0
    if pltOpts['xAxisScale']=='log':
       ist=1
    #
    plt.figure(figsize=(10,6))
    ax=plt.gca()
    plt.plot(y[ist:],pceMean[ist:],'-k',lw='1',label=r'$\mathbb{E}_{\mathbf{q}}[$'+pltOpts['qoiLabel']+'$]$')
    plt.plot(y[ist:],pceMean[ist:]+pceCI[ist:],'-',color='powderblue',alpha=0.7)
    if pltOpts['xAxisScale']=='log':
       plt.semilogx(y[ist:],pceMean[ist:]-pceCI[ist:],'-',color='powderblue',alpha=0.7)
    else:
       plt.plot(y[ist:],pceMean[ist:]-pceCI[ist:],'-',color='powderblue',alpha=0.7)
    ax.fill_between(y[ist:],pceMean[ist:]+pceCI[ist:],pceMean[ist:]-pceCI[ist:],
            color='powderblue',alpha=0.4,label=r'$95\%\, {\rm CI}$ for '+pltOpts['qoiLabel'])
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    if chiName=='y+':
       plt.xlabel(r'$\mathbb{E}_{\mathbf{q}}[y^*]$',fontsize=25)
    else:
       plt.xlabel(pltOpts['chiLabel'],fontsize=25)
    plt.ylabel(pltOpts['qoiLabel'],fontsize=25)
    if 'xLim' in pltOpts.keys():
        plt.xlim(pltOpts['xLim'])
    if 'yLim' in pltOpts.keys():
        plt.ylim(pltOpts['yLim'])
    plt.grid(alpha=0.4)
    if pltOpts['legend']=='on':
       plt.legend(loc='best',fontsize=16)
    #save the figure
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(800/float(DPI),400/float(DPI))
    figDir=pltOpts['figDir']
    if not os.path.exists(figDir):
       os.makedirs(figDir)
    plt.savefig(figDir+pltOpts['figName']+'.pdf',bbox_inches='tight')
    plt.show()
#
def plotUQChan_profSobol(chi,S1,S2,S3,S12,S13,S23,p,chiName,pltOpts):
    """
    Plot Sobol indices of a qoi profile in the wall-normal direction. 
    """            
    texLab=texOptions_tChan.texLabel_constructor()  #Tex labels
    pltOpts_default(pltOpts)
    plt.figure(figsize=(7,4))
    ax=plt.gca()
    parLabs=pltOpts['parLabs']
    plt.plot(chi,S1,'-',color='steelblue',lw=2,label=parLabs[0])
    plt.plot(chi,S2,'-',color='indianred',lw=2,label=parLabs[1])
    if p==3:       
       plt.plot(chi,S3,'-k',lw=2,label=parLabs[2])
    if pltOpts['xLogScale']:
       plt.semilogx(chi,S12,'--g',lw=2,label=parLabs[0]+parLabs[1])
    else:
       plt.plot(chi,S12,'--g',lw=2,label=parLabs[0]+parLabs[1])
    if p==3:
       plt.plot(chi,S13,'--m',lw=2,label=parLabs[0]+parLabs[2])
       plt.plot(chi,S23,'--c',lw=2,label=parLabs[1]+parLabs[2])
    plt.grid()
    if pltOpts['legend']=='on':
       plt.legend(loc='best',fontsize=16)
    if chiName=='y+':
       plt.xlabel(r'$\mathbb{E}_{\mathbf{q}}[y^*]$',fontsize=25)
    else:
       plt.xlabel(texLabel(chiName),fontsize=25)
    plt.ylabel('Sobol Indices',fontsize=25)
    plt.xticks(fontsize=pltOpts['xTicks_fs'])
    plt.yticks(fontsize=pltOpts['yTicks_fs'])
    if 'xLim' in pltOpts.keys():
        plt.xlim(pltOpts['xLim'])
    if 'yLim' in pltOpts.keys():
        plt.ylim(pltOpts['yLim'])
    plt.ylim([0.0,1.0])
    #save the figure
    fig = plt.gcf()
    if 'figDir' in pltOpts.keys():
       figDir=pltOpts['figDir']
       if not os.path.exists(figDir):
          os.makedirs(figDir)
    if 'figName' in pltOpts.keys():
       figName=pltOpts['figName']
       figSave=figDir+figName
    figOut=figDir+figName
    DPI = fig.get_dpi()
    fig.set_size_inches(pltOpts['figSize'][0]/float(DPI),pltOpts['figSize'][1]/float(DPI))
    plt.savefig(figOut,bbox_inches='tight')
    plt.show()
#       
