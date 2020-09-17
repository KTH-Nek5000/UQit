"""
Tests for Sampling
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from sampling import trainSample, testSample, LHS_sampling
#
def trainSample_test():
    """
    Test trainSample()
    """
    F1=trainSample(sampleType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=8)
    F2=trainSample(sampleType='GQ',GQdistType='Norm',qInfo=[0.1,0.5],nSamp=15)
    F3=trainSample(sampleType='normRand',qInfo=[0.1,0.5],nSamp=15)
    F4=trainSample(sampleType='unifRand',qInfo=[2,3],nSamp=10)
    F5=trainSample(sampleType='GLL',qInfo=[2,3],nSamp=10)
    #print
    F=F5
    print('sampleType:',F.sampleType)
    print('GQdistType:',F.GQdistType)
    print('qBound',F.qBound)
    print('xiBound',F.xiBound)
    print('xi',F.xi)
    print('q',F.q)
    print('w',F.w)
    #plot
    m_=0.01
    plt.figure(figsize=(12,3))
    plt.plot(F1.q,  m_*np.ones(F1.q.size),'ob',mfc='none',label='F1.q')
    plt.plot(F2.q,2*m_*np.ones(F2.q.size),'pr',mfc='none',label='F2.q')
    plt.plot(F3.q,3*m_*np.ones(F3.q.size),'sg',mfc='none',label='F3.q')
    plt.plot(F4.q,4*m_*np.ones(F4.q.size),'xm',mfc='none',label='F4.q')
    plt.plot(F5.q,5*m_*np.ones(F5.q.size),'pk',mfc='none',label='F5.q')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

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
    plt.figure(figsize=(12,3))
    m_=0.01
    plt.plot(F1.q,  m_*np.ones(F1.q.size),'ob',mfc='none',label='F1.q')
    plt.plot(F2.q,2*m_*np.ones(F2.q.size),'xr',mfc='none',label='F2.q')
    plt.plot(F3.q,3*m_*np.ones(F3.q.size),'sg',mfc='none',label='F3.q')
    plt.plot(F4.q,4*m_*np.ones(F4.q.size),'+c',mfc='none',label='F4.q')
    plt.plot(F5.q,5*m_*np.ones(F5.q.size),'pk',mfc='none',label='F5.q')
    plt.plot(F6.q,6*m_*np.ones(F6.q.size),'^m',mfc='none',label='F6.q')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()
#
def LHS_sampling_test():
    """
    Test LHS_sampling()
    """
    q=LHS_sampling(100,[[2,3],[-1,4]])
    plt.figure(figsize=(5,5))
    plt.plot(q[:,0],q[:,1],'ob')
    plt.xlabel('q1',fontsize=15)
    plt.ylabel('q2',fontsize=15)
    plt.grid(alpha=0.3)
    plt.show()
#
