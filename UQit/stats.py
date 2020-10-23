#################################
#     Statistical Tools
#################################
#
import os
import sys
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import UQit.write as writeUQ
#
#
def pdfFit_uniVar(f,doPlot,pwOpts):
    """
    Fits a PDF to samples `f` and plots both histogram and the fitted PDF. 
    As an option, the plots and data can be saved on disk.
    
    Args:
      `f`: 1D numpy array of size n 
         Samples
      `doPlot`: bool      
         Whether or not plotting the PDF
      `pwOpts`: dict (optional) 
         Options for plotting and dumping the data with the following keys:
           * 'figDir': string, Directory to save the figure and dump the data
           * 'figName': string, Name of the figure
           * 'header': string, the header of the dumped file
           * 'iLoc': int, After converting to string will be added to the `figName`
    """
    if f.ndim>1:
       print('Note: input f to pdfFit_uniVar(f) must be a 1D numpy array. We reshape f!')
       nTot=1
       for i in range(f.ndim):
           nTot*=f.shape[i]           
       f=np.reshape(f,nTot)     
    #fit kde
    kde = sm.nonparametric.KDEUnivariate(f)
    kde.fit()
    #plot 
    if doPlot:
       plt.figure(figsize=(10,4));
       ax=plt.gca();
       plt.plot(kde.support,kde.density,'-r',lw=2)
       binsNum='auto' #70
       BIN=plt.hist(f,bins=binsNum,density=True,color='steelblue',alpha=0.4,edgecolor='b')
       plt.xticks(fontsize=15)
       plt.yticks(fontsize=15)
       plt.grid(alpha=0.4)
       if pwOpts:    #if not empty
          #Dump the data of the PDf          
          if pwOpts['figDir']:
             figDir=pwOpts['figDir']       
             wrtDir=figDir+'/dumpData/'
          if pwOpts['figName']:   
             outName=pwOpts['figName']
          if pwOpts['iLoc']:
             outName+='_'+str(pwOpts['iLoc'])
          if not os.path.exists(figDir):
             os.makedirs(figDir)
          if not os.path.exists(wrtDir):
             os.makedirs(wrtDir)
          F1=open(wrtDir+outName+'.dat','w')
          if pwOpts['header']:
             F1.write('# '+pwOpts['header']+'\n') 
             F1.write('# kde.support \t\t kde.density \n')
             F1.write('# '+writeUQ.printRepeated('-',50)+'\n')
             for i in range(len(kde.support)):
                 F1.write('%g \t %g \n' %(kde.support[i],kde.density[i]))
             F1.write('# '+writeUQ.printRepeated('-',50)+'\n')
             F1.write('# bin.support \t\t bin.density \n')
             F1.write('# '+writeUQ.printRepeated('-',50)+'\n')
             for i in range(len(BIN[0])):
                 F1.write('%g \t %g \n' %(BIN[1][i],BIN[0][i]))
          #Save the figure of the PDF
          fig = plt.gcf()
          DPI = fig.get_dpi()
          fig.set_size_inches(800/float(DPI),400/float(DPI))
          plt.savefig(figDir+outName+'.pdf',bbox_inches='tight')       
       else:
          plt.show()
    return kde

def pdfPredict_uniVar(f,fTest,doPlot):
    """
    Evaluates the continuous PDF fitted to `f` at `fTest`. 

    Args:
      `f`: 1D numpy array

      `fTest`: List of length m
    
    Returns:
      `pdfPred`: 1D numpy array of size m             
    """
    #Fit the PDF to f
    kde=pdfFit_uniVar(f,doPlot,{})
    #Evaluate the PDF at f0
    pdfPred=[]
    for i in range(len(fTest)):
        pdfPred.append(kde.evaluate(fTest[i])[0])
    pdfPred=np.asarray(pdfPred)
    return pdfPred
# 
