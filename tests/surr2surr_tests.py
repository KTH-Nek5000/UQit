"""
Tests for surr2surr
"""
#
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from surr2surr import lagIntAtGQs
from pce import pce, pceEval
from lagInt import lagInt
import reshaper
import sampling
import analyticTestFuncs
#
#
def lagIntAtGQs_1d_test():
    """
    lagIntAtGQs test for 1D parameter
    """
    #------ SETTINGS --------------------
    #space 1
    nSampMod1=[7]        #number of samples in PCE1
    space1=[[-0.5,2.]]   #admissible space of param in PCE1
    sampleType1='GLL'    #see trainSample class in sampling.py
    #space 2                    
    distType2=['Unif']      #distribution type of the RV
    nSampMod2=[5]        #number of samples in PCE2
    space2=[[0.0,1.5]]   #admissible space of param in PCE2
    #model function
    fType='type1'   #Type of simulator
    #test samples
    nTest=100   #number of test samples
    #------------------------------------
    #(1) Generates samples from SpaceMod1
    qInfo_=space1[0]   
    samps1=sampling.trainSample(sampleType=sampleType1,qInfo=qInfo_,nSamp=nSampMod1[0])
    q1=samps1.q
    xi1=samps1.xi
    qBound1=samps1.qBound
    #Evaluate the simulator at samples1
    fEx=analyticTestFuncs.fEx1D(q1,fType,qInfo_)
    fVal1=fEx.val
    #(2) Lagrange interpolation from samples 1 to GQ nodes on space 2
    q2,xi2,fVal2=lagIntAtGQs(fVal1,[q1],space1,nSampMod2,space2,distType2)
    #(3) Construct a PCE over space 2 using the GQ nodes
    pceDict={'p':1,'sampleType':'GQ','pceSolveMethod':'Projection','distType':distType2}    
    pce2=pce(fVal=fVal2,xi=xi2[:,None],pceDict=pceDict)
    #(4) Evaluate the surrogates: Lagrange interpolation over space 1
    #                             PCE over space 2
    testSamps1=sampling.testSample(sampleType='unifSpaced',qBound=space1[0],nSamp=nTest)
    qTest1=testSamps1.q
    fTest1_ex=analyticTestFuncs.fEx1D(qTest1,fType,space1).val
    fTest1=lagInt(fNodes=fVal1,qNodes=[q1],qTest=[qTest1]).val
    #
    testSamps2=sampling.testSample(sampleType='unifSpaced',qBound=space2[0],nSamp=nTest)
    qTest2=testSamps2.q
    xiTest2=testSamps2.xi
    pcePred_=pceEval(coefs=pce2.coefs,xi=[xiTest2],distType=distType2)
    fTest2=pcePred_.pceVal
    #(5) Plot
    plt.figure(figsize=(15,8))
    plt.plot(qTest1,fTest1_ex,'--k',lw=2,label=r'Exact $f(q)$')
    plt.plot(q1,fVal1,'ob',markersize=8,label='Original samples over space1')
    plt.plot(qTest1,fTest1,'-b',lw=2,label='Lagrange Int. over space 1')
    plt.plot(q2,fVal2,'sr',markersize=8,label='GQ samples over space2')
    plt.plot(qTest2,fTest2,'-r',lw=2,label='PCE over space 2')
    plt.xlabel(r'$q$',fontsize=26)
    plt.ylabel(r'$f(q)$',fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(alpha=0.4)
    plt.legend(loc='best',fontsize=20)
    plt.show()
#
def lagIntAtGQs_2d_test():
    """
       Test pce2pce_GQ(...) for 2D uncertain parameter space
    """
    #------ SETTINGS ----------------------------------------------------
    #Space 1
    nSamp1=[6,10]        #number of samples in PCE1, parameter 1,2
    space1=[[-2,1.5],   #admissible space of PCE1 (both parameters)
             [-3,2.5]]
    sampleType1=['GLL','unifRand']    #see trainSample class in sampling.py
    #Space 2
    nSamp2=[4,5]        #number of samples in PCE2, parameter 1,2
    space2=[[-0.5,1],   #admissible space of PCEw (both parameters)
             [-2.,1.5]]
    #Test samples
    nTest=[100,101]   #number of test samples of parameter 1,2
    #model function
    fType='type1'   #Type of simulator
    #---------------------------------------------------------------------
    p=2
    distType2=['Unif','Unif']
    #(1) Generate samples from space 1
    q1=[]
    for i in range(p):
        q1_=sampling.trainSample(sampleType=sampleType1[i],qInfo=space1[i],nSamp=nSamp1[i])
        space1[i]=[min(q1_.q),max(q1_.q)]   #correction for uniform samples (otherwise contours are not plotted properly)
        q1.append(q1_.q)
    #Response values at the GL points
    fVal1=analyticTestFuncs.fEx2D(q1[0],q1[1],fType,'tensorProd').val
    #(2) Lagrange interpolation from samples 1 to GQ nodes on space 2
    q2,xi2,fVal2=lagIntAtGQs(fVal1,q1,space1,nSamp2,space2,distType2)
    #(3) Construct a PCE on space 2
    pceDict={'p':p,'sampleType':'GQ','pceSolveMethod':'Projection','truncMethod':'TP',
             'distType':distType2}
    pce2=pce(fVal=fVal2,xi=xi2,pceDict=pceDict,nQList=nSamp2)
    #(4) Evaluate the surrogates: Lagrange interpolation over space 1
    #                             PCE over space 2
    #test samples
    qTest1=[]
    xiTest2=[]
    qTest2=[]
    for i in range(p):
        testSamps1=sampling.testSample('unifSpaced',qBound=space1[i],nSamp=nTest[i])
        qTest1.append(testSamps1.q)
        testSamps2=sampling.testSample('unifSpaced',GQdistType=distType2[i],qBound=space2[i],nSamp=nTest[i])
        xiTest2.append(testSamps2.xi)
        qTest2.append(testSamps2.q)
    #evaluation
    #space 1
    fTest1_ex=analyticTestFuncs.fEx2D(qTest1[0],qTest1[1],fType,'tensorProd').val
    fTest1=lagInt(fNodes=fVal1,qNodes=q1,qTest=qTest1,liDict={'testRule':'tensorProd'}).val
    #space 2
    pceEval2=pceEval(coefs=pce2.coefs,xi=xiTest2,distType=distType2,kSet=pce2.kSet)
    fTest2=pceEval2.pceVal
    #(5) 2d contour plots
    plt.figure(figsize=(20,8))
    plt.subplot(1,3,1)
    ax=plt.gca()
    fTest_Grid=fTest1_ex.reshape(nTest,order='F').T
    CS1 = plt.contour(qTest1[0],qTest1[1],fTest_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact response surface over space 1')
    #
    plt.subplot(1,3,2)
    ax=plt.gca()
    fTest1_Grid=fTest1.reshape(nTest,order='F').T
    CS2 = plt.contour(qTest1[0],qTest1[1],fTest1_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    q1Grid=reshaper.vecs2grid(q1)
    plt.plot(q1Grid[:,0],q1Grid[:,1],'ob',markersize=6)
    q2_=reshaper.vecs2grid(q2)
    plt.plot(q2_[:,0],q2_[:,1],'sr',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response surface by Lagrange Int.\n over space-1 using blue circles')
    #
    plt.subplot(1,3,3)
    ax=plt.gca()
    fTest2_Grid=fTest2.reshape(nTest,order='F').T
    CS3 = plt.contour(qTest2[0],qTest2[1],fTest2_Grid,20)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(q2_[:,0],q2_[:,1],'sr',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response surface by PCE over space-2 \n using red squares')
    plt.xlim(space1[0][:])
    plt.ylim(space1[1][:])
    plt.show()
#    
