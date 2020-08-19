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
    def __init__(self,sampType='',GQdistType='',qInfo=[],nSamp=0):
        self.info()
        self.sampType=sampType
        self.GQdistType=GQdistType
        self.check()
        self.qInfo=qInfo
        self.nSamp=nSamp
        self.w=[[]]*self.nSamp
        self.genSamples()

    def info(self):
        sampTypeList=['GQ','unifSpaced','unifRand','normRand','Clenshaw',\
                      'Clenshaw-Curtis']
        GQdistList=['Unif','Norm'] #list of available distributions for gpce
        self.sampTypeList=sampTypeList
        self.GQdistList=GQdistList

    def check(self):
        if self.sampType not in self.sampTypeList:
           print('#ERROR @ parSample: Invalid sampType') 
           print('Available sampType:',self.samptypeList)
        if self.sampType=='GQ' and self.GQdistType not in self.GQdistList:
           print('#ERROR @ parSample: Invalid GQdistType') 
           print('Available GQsampType:',self.GQdistList)

    def genSamples(self):       
        n=self.nSamp
        if self.sampType=='GQ' and self.GQdistType in self.GQdistList:
           self.gqPtsWts() 
        elif self.sampType=='normRand':
           self.xi=np.random.normal(size=n)
           self.xiBound=[min(self.xi),max(self.xi)]
           self.mean=self.qInfo[0]
           self.sdev=self.qInfo[1]
           self.xi2q_map()         
        else:    
           if self.sampType=='unifSpaced':
              self.xiBound=[0,1]
              self.xi=np.linspace(0,1,n)
           elif self.sampType=='unifRand':
              self.xiBound=[0,1]
              self.xi=np.random.rand(n)
           elif self.sampType=='Clenshaw':
              self.xiBound=[-1,1]
              self.xi=nodes.Clenshaw_pts(n)
           elif self.sampType=='Clenshaw-Curtis':
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
        if (self.sampType=='GQ' and self.GQdistType=='Norm') or \
            self.sampType=='normRand':
           self.q=self.qInfo[0]+self.qInfo[1]*xi_     
           self.qBound=[min(self.q),max(self.q)]
        else:
           xiBound_=self.xiBound
           qBound_=self.qBound
           self.q=(xi_-xiBound_[0])/(xiBound_[1]-xiBound_[0])*\
                  (qBound_[1]-qBound_[0])+qBound_[0]
#
#
# Test
def test():
    #F=parSample(sampType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=10)
    F=parSample(sampType='normRand',qInfo=[2,3],nSamp=10)
    print('sampType:',F.sampType)
    print('GQdistType:',F.GQdistType)
    print('qBound',F.qBound)
    print('xiBound',F.xiBound)
    print('xi',F.xi)
    print('q',F.q)
    print('w',F.w)


