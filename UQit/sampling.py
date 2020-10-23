#############################################
#     Sampling from parameter space
#############################################
#--------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------
#
import sys
import os
import numpy as np
import math as mt
import UQit.nodes as nodes
import UQit.pce as pce
#
class trainSample:
    R"""
    Generating training samples from a 1D paramter space using different methods.
    Samples of `xi` are drawn from the mapped space Gamma and are then mapped to the parameter space Q.

    Args:
      `sampleType`: string
         Sample type, chosen from the following list:
           * 'GQ': Gauss-Quadrature nodes 
           * 'GLL': Gauss-Lobatto-Legendre nodes
           * 'unifSpaced': Uniformly-spaced
           * 'unifRand': Uniformly distributed random
           * 'normRand': Gaussian distributed random
           * 'Clenshaw': Clenshaw points
           * 'Clenshaw-Curtis': Clenshaw-Curtis points
      `GQdistType`: string (optional)
         Specifies type of gPCE standard distribution if sampleType=='GQ'
           * 'Unif': Uniform distribution, Gamma=[-1,1]            
           * 'Norm': Gaussian distribution, Gamma=[-\infty,\infty]            
      `qInfo`: List of length 2 (optional)
         Information about the parameter range or distribution.
           * If `q` is Gaussian ('Norm' or 'normRand') => qInfo=[mean,sdev]
           * Otherwise, `qInfo`=[min(q),max(q)]=admissible range of `q`
      `nSamp`: Integer
         Number of samples to draw

    Attributes:     
      `xi`: 1D numpy array of size nSamp
         Samples drawn from the mapped space Gamma    
      `xiBound`: List of length 2
         Admissible range of `xi`
      `q`: 1D numpy array of size `nSamp`
         Samples over the parameter space Q    
      `qBound`: List of length 2
         Admissible range of `q`
      `w`: 1D numpy array of size `nSamp`
         Weights in Gauss-Quadrature rule only if sampleType=='GQ'         
    
    Examples:
      ts1=trainSample(sampleType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=10)
      ts2=trainSample(sampleType='NormRand',qInfo=[2,3],nSamp=10)
      ts3=trainSample(sampleType='GLL',qInfo=[2,3],nSamp=10)
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
        sampleTypeList=['GQ','GLL','unifSpaced','unifRand','normRand','Clenshaw',\
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
           if self.sampleType=='GLL':
              self.xiBound=[-1,1]
              xi_,w_=nodes.gllPts(n) 
              self.xi=xi_
              self.w=w_
           if self.sampleType=='unifSpaced':
              xiBound_=[0,1]
              self.xiBound=xiBound_
              self.xi=np.linspace(xiBound_[0],xiBound_[1],n)
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
        Gauss-Quadrature nodes and weights according to the gPCE rule
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
        Linearly map xi in Gamma to q in Q
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
class testSample:
    R"""
    Generating test samples from a 1D paramter space using different methods.
    Samples of q in parameter space Q are drawn and then mapped to xi in the mapped space Gamma.

    Args:
      `sampleType`: string 
         Type of sample, chosen from the following list:
           * 'GLL': Gauss-Lobatto-Legendre nodes
           * 'unifSpaced': Uniformly-spaced
           * 'unifRand': Uniformly distributed random
           * 'normRand': Gaussian distributed random
      `GQdistType`: string
         Type of standard distribution in gPCE; default is 'Unif'
           * 'Unif': Uniform distribution, Gamma=[-1,1]            
           * 'Norm': Gaussian distribution, Gamma=[-\infty,\infty]            
      `qInfo`: List of length 2 (optional)         
         qInfo=[mean,sdev] only if GQdistType=='Norm'
      `qBound`: List of length 2 
         Admissible range of `q`
      `nSamp`: int
         Number of samples to draw
    
    Attributes:
      `xi`: 1D numpy array of size `nSamp`
         Samples on the mapped space Gamma    
      `xiBound`: List of length 2
         Admissible range of `xi`
      `q`: 1D numpy array of size `nSamp`
         Samples `q` from the mapped space Q    
      `qBound`: List of length 2 
         Admissible range of `q`. It will be the same as the argument `qBound` if GQdistType=='Unif'

    Examples:
      ts1=testSample(sampleType='unifRand',GQdistType='Unif',qBound=[-1,3],nSamp=10)
      ts2=testSample(sampleType='unifRand',qBound=[-1,3],nSamp=10)
      ts3=testSample(sampleType='unifSpaced',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10)
      ts4=testSample(sampleType='normRand',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10)
      ts5=testSample(sampleType='unifSpaced',GQdistType='Unif',qBound=[-1,3],nSamp=10)
      ts6=testSample(sampleType='GLL',qBound=[-1,3],nSamp=10)
    """
    def __init__(self,sampleType,qBound,nSamp,GQdistType='Unif',qInfo=[]):
        self.info()
        self.sampleType=sampleType
        self.GQdistType=GQdistType
        self.qInfo=qInfo
        self.check()
        self.qBound=qBound
        self.nSamp=nSamp
        self.genTestSamples()

    def info(self):
        sampleTypeList=['GLL','unifSpaced','unifRand','normRand']
        GQdistList=['Unif','Norm'] #list of available distributions for gpce
        self.sampleTypeList=sampleTypeList
        self.GQdistList=GQdistList

    def check(self):
        if self.sampleType not in self.sampleTypeList:
           raise KeyError('#ERROR @ testSample: Invalid sampleType! Choose from'\
                   ,self.sampleTypeList)
        if self.GQdistType=='Norm' and len(self.qInfo)==0:
           raise KeyError("#ERROR @ testSample: qInfo is mandatory for GQdistType='Norm'")

    def genTestSamples(self):       
        n=self.nSamp
        if self.GQdistType=='Unif':
           self.xiBound=[-1,1]

        if self.sampleType=='unifSpaced':
           q_=np.linspace(self.qBound[0],self.qBound[1],n) 
        elif self.sampleType=='GLL':
             self.xiBound=[-1,1]
             xi_,w_=nodes.gllPts(n) 
             q_=xi_*(self.qBound[1]-self.qBound[0])+self.qBound[0]
        elif self.sampleType=='unifRand':
           if self.GQdistType!='Unif': 
              raise ValueError("#ERROR @ testSample: sampleType 'unifRand' should be with GQdistType 'Unif' or ''.")
           q_=np.random.rand(n)*(self.qBound[1]-self.qBound[0])+self.qBound[0]
           q_=np.sort(q_)
        elif self.sampleType=='normRand':
           if self.GQdistType!='Norm': 
               raise ValueError("#ERROR @ testSample: sampleType 'normRand' should be with\
                       GQdistType 'Norm'.")
           else:
              xi_=np.random.normal(size=n)
              xi_=np.sort(xi_)
              self.xi=xi_
              q_=self.qInfo[0]+xi_*self.qInfo[1]
        self.q=q_
        self.q2xi_map()      
        
    def q2xi_map(self):
        """
        Linearly map q in Q to xi in Gamma
        """
        q_=self.q
        qBound_=self.qBound
        if self.GQdistType=='Norm':   
           xi_=(q_-self.qInfo[0])/self.qInfo[1]
           self.xiBound=[min(xi_),max(xi_)]
        else:
           xi_=(self.xiBound[1]-self.xiBound[0])*(q_-qBound_[0])\
               /(qBound_[1]-qBound_[0])+self.xiBound[0]
        self.xi=xi_    
#
def LHS_sampling(n,xBound):
    """
        LHS (Latin Hypercube) sampler from a p-D random variable distributed uniformly.
        Credits: https://zmurchok.github.io/2019/03/15/Latin-Hypercube-Sampling.html

        Args:
          `n`: int
             Number of samples to be taken
          `xBound`: list of length p
             =[[min(x1),max(x1)],...[min(xp),max(xp)]], where [min(xi),max(xi)] specifies
             the range of the i-th parameter

        Returns:
          `x`: 2D numpy array of size (n,p)
             Samples taken from the p-D space with ranges `xBound`
    """
    p=len(xBound)
    x = np.random.uniform(size=[n,p])
    for i in range(p):
        x_ = (np.argsort(x[:,i])+0.5)/float(n)
        x[:,i]=x_*(xBound[i][1]-xBound[i][0])+xBound[i][0]
    return x
#
