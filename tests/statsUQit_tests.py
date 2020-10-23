"""
Tests for stats
"""
#
import os
import sys
import numpy as np
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
import UQit.stats as statsUQit
#
#
def pdf_uniVar_test():
    """
        For a set of randomly generated data, plot histogram and fitted PDF.
        Also, predict the PDF value at an arbitrary points.
    """
    def bimodal_samples(n):
        """
           Samples from a bimodal distribution
           The script in this function is taken from
           https://www.statsmodels.org/stable/examples/notebooks/generated/kernel_density.html
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
    pdf_test=statsUQit.pdfPredict_uniVar(y,yTest,doPlot=True)
    print('yTest= ',yTest)
    print('PDF of y evaluated at yTest points: ',pdf_test)

