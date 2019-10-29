#######################################################
#      Fit PDF to data
#######################################################
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

#////////////////////////////////
def pdfFit_uniVar(f,doPlot):
    """
        Fit a PDF to data f and plot both histogram and continuous PDF. 
        f: 1d(=uniVar) numpy array of size n 
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
       plt.hist(f,bins='auto',density=True,color='steelblue',alpha=0.4,edgecolor='b')
       plt.xticks(fontsize=15)
       plt.yticks(fontsize=15)
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
    kde=pdfFit_uniVar(f,doPlot)
    #Evaluate the PDF at f0
    pdfPred=[]
    for i in range(len(fTest)):
        pdfPred.append(kde.evaluate(fTest[i])[0])
    pdfPred=np.asarray(pdfPred)
    return pdfPred
  
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
    
