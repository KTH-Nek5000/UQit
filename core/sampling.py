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
sys.path.append(os.getenv("UQit"))
import nodes
import pce
#
def LHS_sampling(n,range_):
    """
        LHS (Latin Hypercube) sampler from a p-D random variable distributed uniformly
        credit: https://zmurchok.github.io/2019/03/15/Latin-Hypercube-Sampling.html
        Inputs:
              n: number of samples
              range_: Admissible orange of the samples,list of size p, =[[x1L,x1U],...[xpL,xpU]]
        Outputs:
              x: n-by-p numpy array, x\in range_
    """
    p=len(range_)
    x = np.random.uniform(size=[n,p])
    for i in range(0,p):
        x_ = (np.argsort(x[:,i])+0.5)/float(n)
        x[:,i]=x_*(range_[i][1]-range_[i][0])+range_[i][0]
    return x
#
def sampler_1d(range_,nSamp,sampType):
    """
    Generating samples from a 1D parameter space
    Inputs:
       range_: list of length 2, admissible range of parameters
       nSamp: Number of samples
       sampType: The method to drawing the samples (nodes)
                 'random', 'uniform', 'GL', 'Clenshaw', 'Clenshaw-Curtis'
    Outputs:
       qNodes: samples of size nSamp over range_ 
    """
    p=len(range_)
    xi_len=1.
    xi0=0.
    if sampType=='random':
       xi=np.random.uniform(0,1,size=[nSamp-2])
       xi=np.concatenate([[0],xi,[1]])
    elif sampType =='uniform':
       xi=np.linspace(0,1,nSamp)
    elif sampType =='GL':
       xi,wXI=pce.gqPtsWts(nSamp,'Unif')  #on [-1,1]
       xi_len=2.
       xi0=-1.
    elif sampType=='Clenshaw':
       xi=nodes.Clenshaw_pts(nSamp)
       xi_len=2.
       xi0=-1.
    elif sampType=='Clenshaw-Curtis':
       l_=1+int(mt.log(nSamp-1)/mt.log(2))
       xi=nodes.ClenshawCurtis_pts(l_)
    else:
       print('ERROR in sampler_1d(): Invalid sampType was chosen!')
    #map from reference range to actual range
    qNodes=(range_[1]-range_[0])*(xi-xi0)/xi_len+range_[0]
    return qNodes

class trainSample:
    R"""
    Generating training samples from 1D paramter space using different methods.
    Samples of \xi\in\Gamma (mapped space) are drawn which then mapped to q\in Q

    Parameters
    ----------
    sampleType: string
        Type of sample:
        'GQ': Gauss-Quadrature nodes 
        'GLL': Gauss-Lobatto-Lgendre nodes
        'unifSpaced': Uniformly-spaced
        'unifRand': Uniformly distributed random
        'normRand': Gaussian distributed random
        'Clenshaw': Clenshaw points
        'Clenshaw-Curtis': Clenshaw-Curtis points
    GQdistType: string
        Type of standard distribution in gPCE; Only needed if sampleType:'GQ'
        'Unif': Uniform distribution, Gamma=[-1,1]            
        'Norm': Gaussian distribution, Gamma=[-\infty,\infty]            
    qInfo: List of length 2
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

class testSample:
    R"""
    Generating test samples from 1D paramter space using different methods.
    Samples of q\in Q are drawn which then mapped to \xi\in\Gamma (mapped space)

    Parameters
    ----------
    sampleType: string
        Type of sample:
        'GLL': Gauss-Lobatto-Lgendre nodes
        'unifSpaced': Uniformly-spaced
        'unifRand': Uniformly distributed random
        'normRand': Gaussian distributed random
    GQdistType: string
        Type of standard distribution in gPCE; default='Unif'
        'Unif': Uniform distribution, Gamma=[-1,1]            
        'Norm': Gaussian distribution, Gamma=[-\infty,\infty]            
    qInfo: List of length 2
        Information on the parameter.
        If q is Gaussian ('Norm') => qInfo=[mean,sdev]
        Otherwise, qInfo=[] and providing it is optional
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
    def __init__(self,sampleType='',GQdistType='Unif',qInfo=[],qBound=[],nSamp=0):
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
        if self.sampleType=='GQ' and self.GQdistType not in self.GQdistList:
           raise KeyError('#ERROR @ testSample: Invalid GQdistType! Choose from'\
                   ,self.GQdistList)
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
        Linearly map q\in Q to xi\in\Gamma
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
#
# Tests
#
def trainSample_test():
    """
    Test trainSample()
    """
    #F=trainSample(sampleType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=10)
    #F=trainSample(sampleType='NormRand',qInfo=[2,3],nSamp=10)
    F=trainSample(sampleType='GLL',qInfo=[2,3],nSamp=10)
    print('sampleType:',F.sampleType)
    print('GQdistType:',F.GQdistType)
    print('qBound',F.qBound)
    print('xiBound',F.xiBound)
    print('xi',F.xi)
    print('q',F.q)
    print('w',F.w)

import matplotlib.pyplot as plt
def testSample_test():
    """
    Test testSample()
    """
    F1=testSample(sampleType='unifRand',GQdistType='Unif',qBound=[-1,3],nSamp=10)
    F2=testSample(sampleType='unifRand',qBound=[-1,3],nSamp=10)
    F3=testSample(sampleType='normRand',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10)
    F4=testSample(sampleType='unifSpaced',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10)
    F5=testSample(sampleType='unifSpaced',GQdistType='Unif',qBound=[-1,3],nSamp=10)
    F6=testSample(sampleType='GLL',qBound=[-1,3],nSamp=10)
    print('sampleType:',F1.sampleType)
    print('GQdistType:',F1.GQdistType)
    print('qBound',F1.qBound)
    print('xiBound',F1.xiBound)
    print('xi',F1.xi)
    print('q',F1.q)
    plt.plot(F1.q,'ob')
    plt.plot(F2.q,'xr')
    plt.plot(F3.q,'sg')
    plt.plot(F4.q,'+c')
    plt.plot(F5.q,'pk')
    plt.plot(F6.q,'^m')
    plt.show()
