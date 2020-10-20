"""
Tests for lagInt
"""
#
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from lagInt import lagInt, lagInt_Quads2Line
import analyticTestFuncs
import sampling
import pce
import reshaper
#
#
def lagInt_1d_test():
    """
    Test Lagrange inerpolation over a 1D parameter space.
    """
    #----- SETTINGS -------------------
    nNodes=15         #number of training samples (nodes)
    qBound=[-1,3]     #range over which the samples are taken
    nTest=100         #number of test points
    sampType='GLL'    #Type of samples, see trainSample class in samping.py
    fType='type1'     #Type of model function used as simulator
    #----------------------------------
    # Create the training samples and evaluate the simulator at each sample
    samps_=sampling.trainSample(sampleType=sampType,qInfo=qBound,nSamp=nNodes)
    qNodes=samps_.q
    fNodes=analyticTestFuncs.fEx1D(qNodes,fType,qBound).val
    # Generate the test samples
    qTestFull=np.linspace(qBound[0],qBound[1],nTest)
    qTest=np.linspace(min(qNodes),max(qNodes),nTest)
    # Construct the Lagrange interpolation and evaluate it at the test points
    fInterpTest=lagInt(fNodes=fNodes,qNodes=[qNodes],qTest=[qTest]).val
    # Plot
    fTestFull=analyticTestFuncs.fEx1D(qTestFull,fType,qBound).val
    plt.figure(figsize=(12,7))
    plt.plot(qTestFull,fTestFull,'--r',lw=2,label='Exact f(q)')
    plt.plot(qTest,fInterpTest,'-b',lw=2,label='f(q) by Lagrange Interpolation')
    plt.plot(qNodes,fNodes,'oc',markersize='8',label='Nodes')
    plt.legend(loc='best',fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.xlabel(r'$q$',fontsize=26)
    plt.ylabel(r'$f(q)$',fontsize=26)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1000/float(DPI),500/float(DPI))
    plt.savefig('../testFigs/lag1d4'+'.png',bbox_inches='tight')
    plt.show()
#
def lagInt_2d_test():
    """
    Test Lagrange inerpolation over a 2D parameter space.
    """
    #----- SETTINGS --------------------------------------------------------------
    nNodes=[5,4]          #number of  training samples nodes in space of parameters q1, q2
    sampType=['GLL',      #Method of drawing samples for q1, q2
              'unifSpaced']
    qBound=[[-0.75,1.5],  # admissible range of parameters
            [-0.5 ,2.5]]

    # Settings of the exact response surface
    domRange=[[-2,2], #domain range for q1, q2
              [-3,3]]
    nTest=[100,101] #number of test samples 
    #-----------------------------------------------------------------------------
    p=len(nNodes)
    # Create the training samples over each parameter space
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)
    # Evaluate the simulator at each joint sample
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd').val
    # Generate the test samples
    qTestList=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTestList.append(qTest_.q)
    # Construct the Lagrange interpolation and evaluate it at the test samples
    fTest=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTestList,liDict={'testRule':'tensorProd'}).val
    # Evaluate the exact model response over domRange
    qTestFull=[]
    for i in range(p):
        qTestFull_=np.linspace(domRange[i][0],domRange[i][1],nTest[i])  
        qTestFull.append(qTestFull_)
    fTestFull=analyticTestFuncs.fEx2D(qTestFull[0],qTestFull[1],'type1','tensorProd').val
    fTestFullGrid=fTestFull.reshape((nTest[0],nTest[1]),order='F').T
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F').T
    # Plots
    plt.figure(figsize=(16,8));
    plt.subplot(1,2,1)
    ax=plt.gca()
    CS1 = plt.contour(qTestFull[0],qTestFull[1],fTestFullGrid,35)
    plt.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    qNodesGrid=reshaper.vecs2grid(qNodes)  
    plt.plot(qNodesGrid[:,0],qNodesGrid[:,1],'o',color='r',markersize=6)
    plt.xlabel(r'$q_1$',fontsize=25);plt.ylabel(r'$q_2$',fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('Exact Response Surface')
    plt.subplot(1,2,2)
    ax=plt.gca()
    CS2 = plt.contour(qTestList[0],qTestList[1],fTestGrid,20)
    plt.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qNodesGrid[:,0],qNodesGrid[:,1],'o',color='r',markersize=6)
    plt.xlabel(r'$q_1$',fontsize=25);plt.ylabel(r'$q_2$',fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('Response Surface by Lagrange Interpolation')
    plt.xlim(domRange[0])
    plt.ylim(domRange[1])
    plt.show()
#
def lagInt_3d_test():
    """
    Test Lagrange inerpolation over a 3D parameter space.
    """
    #----- SETTINGS -------------------
    nNodes=[8,7,6]         #number of training samples for q1, q2, q3
    sampType=['GLL',       #Type of samples for q1, q2, q3
              'unifSpaced',
              'Clenshaw']
    qBound=[[-0.75,1.5],   #range of parameters q1, q2, q3
            [-0.5 ,2.5],
            [1,3]]
    nTest=[10,11,12]       #number of test samples for q1, q2, q3
    fOpts={'a':7,'b':0.1}  #parameters in Ishigami function
    #----------------------------------
    p=len(nNodes)
    # Generate the training samples
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)
    # Run the simulator at the training samples   
    fNodes=analyticTestFuncs.fEx3D(qNodes[0],qNodes[1],qNodes[2],'Ishigami','tensorProd',fOpts).val
    # Create the test samples and run the simultor at them
    qTest=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTest.append(qTest_.q)
    fTestEx=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',fOpts).val
    # Construct the Lagrange interpolation an evaluate it at the test samples
    fInterp=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTest,liDict={'testRule':'tensorProd'}).val
    # Plot
    plt.figure(figsize=(14,8))
    plt.subplot(2,1,1)
    fInterp_=fInterp.reshape(np.asarray(np.prod(np.asarray(nTest))),order='F')
    plt.plot(fInterp_,'-ob',mfc='none',label='Lagrange Interpolation')
    plt.plot(fTestEx,'--xr',ms=5,label='Exact Value')
    plt.ylabel(r'$f(q_1,q_2,q_3)$',fontsize=18)
    plt.xlabel(r'Test Sample Number',fontsize=14)
    plt.legend(loc='best',fontsize=14)
    plt.grid(alpha=0.4)
    plt.subplot(2,1,2)
    plt.plot(abs(fInterp_-fTestEx),'-sk')
    plt.ylabel(r'$|f_{Interp}(q)-f_{Exact}(q)|$',fontsize=15)
    plt.xlabel(r'Test Sample Number',fontsize=14)
    plt.grid(alpha=0.4)
    plt.show()
#
def lagInt_Quads2Line_test():
    """
    Test lagInt_Quads2Line().
    The test samples of (q1,q2) are generated along a defined line q2=a*q1+b 
    in the admissible space of q1-a2. 
    The training samples are drawn in the usual way, covering the admissible space of q1-q2.
    """
    #----- SETTINGS --------------------------------------------------------------
    nNodes=[9,9]          #number of training samples for q1, q2
    sampType=['GLL',      #type of training samples for q1, q2
              'unifSpaced']
    qBound=[[-0.75,1.5],  #admissible range of q1,q2 
            [-0.8 ,2.5]]  #Note that the line should be confined in this space
    lineDef={'start':[1.4,2.3],    #coordinates of the line's starting point in the q1-q2 plane
             'end':[-0.7,-0.2],    #coordinates of the line's end point in the q1-q2 plane
             'noPtsLine':100       #number of the test samples
            }
    #-----------------------------------------------------------------------------
    p=len(nNodes)
    # Generate the training samples
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)
    # Evaluate the simulator at the training samples
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd').val
    # Construct the lagrange interpolation and evalautes it at the test points over the line
    qLine,fLine=lagInt_Quads2Line(fNodes,qNodes,lineDef)
    # Plots
    plt.figure(figsize=(8,5))
    plt.plot(qLine[0],fLine,'-ob',mfc='none',label='Lagrange Interpolation')
    fLine_ex=analyticTestFuncs.fEx2D(qLine[0],qLine[1],'type1','comp').val #exact response
    plt.plot(qLine[0],fLine_ex,'-xr',label='Exact Value')
    plt.xlabel(r'$q_1$',fontsize=16);
    plt.ylabel('Response',fontsize=14)
    plt.legend(loc='best')
    plt.grid(alpha=0.4)
    plt.show()
#    
