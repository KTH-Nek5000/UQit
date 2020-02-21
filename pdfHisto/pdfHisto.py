#######################################################
#      Fit PDF to data
#######################################################
import os
import sys
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../writeUQ/')
import writeUQ

#////////////////////////////////
def pdfFit_uniVar(f,doPlot,pwOpts):
    """
        Fit a PDF to data f and plot both histogram and continuous PDF. 
        f: 1d(=uniVar) numpy array of size n 
        doPlot=False or True
        pwOpts: (optional) options for plotting and dumping the data
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
       binsNum=70 #'auto'
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
             F1.write('#'+pwOpts['header']+'\n') 
             F1.write('# kde.support \t\t kde.density \n')
             F1.write(writeUQ.printRepeated('-',50)+'\n')
             for i in range(len(kde.support)):
                 F1.write('%g \t %g \n' %(kde.support[i],kde.density[i]))
             F1.write(writeUQ.printRepeated('-',50)+'\n')
             F1.write('# bin.support \t\t bin.density \n')
             F1.write(writeUQ.printRepeated('-',50)+'\n')
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

#///////////////////////////////
def pdfPredict_uniVar(f,fTest,doPlot):
    """
       Evaluate continuous PDF fitted to f at fTest. 
       f: 1D numpy array
       f0: 1D list
    """
    #Fit the PDF to f
    kde=pdfFit_uniVar(f,doPlot,{})
    #Evaluate the PDF at f0
    pdfPred=[]
    for i in range(len(fTest)):
        pdfPred.append(kde.evaluate(fTest[i])[0])
    pdfPred=np.asarray(pdfPred)
    return pdfPred
 
##################
# TEST
##################
#/////////////////////////
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
def pdf_uniVar_test():
    """
        For a set of randomly generated data, plot histogram and fitted PDF. 
        Also, predict the PDF value at an arbitrary points.
    """
    def bimodal_samples(n):
        """
           Samples from a bimodal distribution
           The script in this function is taken from: https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html
        """
        # Location, scale and weight for the two distributions
        dist1_loc, dist1_scale, weight1 = -1 , .4, .3
        dist2_loc, dist2_scale, weight2 = 1 , .5, .7
        # Sample from a mixture of distributions
        f = mixture_rvs(prob=[weight1, weight2], size=n,
                               dist=[stats.norm, stats.norm],
                               kwargs = (dict(loc=dist1_loc, scale=dist1_scale),
                                         dict(loc=dist2_loc, scale=dist2_scale)))
        return f

    y=bimodal_samples(1000)    #observed data to which PDF is fitted
    yTest=-4.0+(9)*np.random.rand(10)   #test data
    pdf_test=pdfPredict_uniVar(y,yTest,doPlot=True)
    print('yTest= ',yTest)
    print('PDF of y evaluated at yTest points: ',pdf_test)
    
