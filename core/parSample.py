#################################################
#  Generating samples from a 1D parameter space
#################################################
#------------------------------------------------
# saleh Rezaeiravesh, salehr@kth.se
#------------------------------------------------
import sys
import os
import numpy as np
import math as mt
sys.path.append(os.getenv("UQit"))
import nodes
#
class parSample:
    R"""
    Generating samples from 1D paramter space using different methods.

    Parameters
    ----------
    sampleType: string
        Type of sample:
        'GQ': Gauss-Quadrature nodes 
        'unifSpaced': Uniformly-spaced
        'unifRand': Uniformly distributed random
        'normRand': Gaussian distributed random
        'Clenshaw': Clenshaw points
        'Clenshaw-Curtis': Clenshaw-Curtis points
    GQdistType: string
        Type of standard distribution in gPCE; Only needed if sampleType:'GQ'
        'Unif': Uniform distribution, Gamma=[-1,1]            
        'Norm': Gaussian distribution, Gamma=[-\infty,\infty]            
    qInfo: list of length 2
        Information on the parameter.
        If q is Gaussian ('Norm' or 'normRand') => qInfo=[mean,sdev]
        Otherwise, qInfo=[min(q),max(q)]=admissible range of q
    nSamp: Integer
        Number of samples to draw
    
    Attributes
    ----------
    xi: 1d numpy array of size nSamp
        Samples \xi from the mapped space \Gamma    
    xiBound: List of length 2
        Admissible range of xi
    q: 1d numpy array of size nSamp
        Samples q from the mapped space Q    
    qBound: List of length 2
        Admissible range of q
    w: 1d numpy array of size nSamp    
       Weights in Gauss-Quadrature rule (only if sampleType='GQ')
    """
    def __init__(self,sampleType='',GQdistType='',qInfo=[],nSamp=0):
        self.info()
        self.sampleType=sampleType
        self.GQdistType=GQdistType
        self.check()
        self.qInfo=qInfo
        self.nSamp=nSamp
        self.w=[[]]*self.nSamp
        self.genSamples()

    def info(self):
        sampleTypeList=['GQ','unifSpaced','unifRand','normRand','Clenshaw',\
                      'Clenshaw-Curtis']
        GQdistList=['Unif','Norm'] #list of available distributions for gpce
        self.sampleTypeList=sampleTypeList
        self.GQdistList=GQdistList

    def check(self):
        if self.sampleType not in self.sampleTypeList:
           raise KeyError('#ERROR @ parSample: Invalid sampleType! Choose from'\
                   ,self.sampleTypeList)
        if self.sampleType=='GQ' and self.GQdistType not in self.GQdistList:
           raise KeyError('#ERROR @ parSample: Invalid GQdistType! Choose from'\
                   ,self.GQdistList)

    def genSamples(self):       
        n=self.nSamp
        if self.sampleType=='GQ' and self.GQdistType in self.GQdistList:
           self.gqPtsWts() 
        elif self.sampleType=='normRand':
           self.xi=np.random.normal(size=n)
           self.xiBound=[min(self.xi),max(self.xi)]
           self.mean=self.qInfo[0]
           self.sdev=self.qInfo[1]
           self.xi2q_map()         
        else:    
           if self.sampleType=='unifSpaced':
              self.xiBound=[0,1]
              self.xi=np.linspace(0,1,n)
           elif self.sampleType=='unifRand':
              self.xiBound=[0,1]
              self.xi=np.random.rand(n)
           elif self.sampleType=='Clenshaw':
              self.xiBound=[-1,1]
              self.xi=nodes.Clenshaw_pts(n)
           elif self.sampleType=='Clenshaw-Curtis':
              self.xiBound=[0,1]
              l_=1+int(mt.log(n-1)/mt.log(2))
              self.xi=nodes.ClenshawCurtis_pts(l_)
           self.qBound=self.qInfo
           self.xi2q_map()

    def gqPtsWts(self):
        """
        Gauss-Quadrature nodes and weights acc. gPCE rule
        """
        n=self.nSamp
        if self.GQdistType=='Unif':
           x=np.polynomial.legendre.leggauss(n)
           self.xi=x[0]
           self.w=x[1]
           self.xiBound=[-1,1]
           self.qBound=self.qInfo 
        elif self.GQdistType=='Norm':
           x=np.polynomial.hermite_e.hermegauss(n)
           self.xi=x[0]
           self.w=x[1]
           self.xiBound=[min(x[0]),max(x[0])]
           self.mean=self.qInfo[0]
           self.sdev=self.qInfo[1]
        self.xi2q_map()

    def xi2q_map(self):
        """
        Linearly map xi\in\Gamma to q\in Q
        """
        xi_=self.xi
        if (self.sampleType=='GQ' and self.GQdistType=='Norm') or \
            self.sampleType=='normRand':
           self.q=self.qInfo[0]+self.qInfo[1]*xi_     
           self.qBound=[min(self.q),max(self.q)]
        else:
           xiBound_=self.xiBound
           qBound_=self.qBound
           self.q=(xi_-xiBound_[0])/(xiBound_[1]-xiBound_[0])*\
                  (qBound_[1]-qBound_[0])+qBound_[0]
#
#
# Tests
#
def parSample_test():
    """
    Test parSample()
    """
    F=parSample(sampleType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=10)
    #F=parSample(sampleType='NormRand',qInfo=[2,3],nSamp=10)
    print('sampleType:',F.sampleType)
    print('GQdistType:',F.GQdistType)
    print('qBound',F.qBound)
    print('xiBound',F.xiBound)
    print('xi',F.xi)
    print('q',F.q)
    print('w',F.w)
