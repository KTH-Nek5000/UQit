"""
Tests for lagInt
"""
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
def lagInt_1d_test():
    """
    Test Lagrange inerpolation over a 1D parameter space.
    """
    #----- SETTINGS -------------------
    nNodes=15         #number of nodes
    qBound=[-1,3]     #range over which nodes are randomly chosen
    nTest=100         #number of test points for plot
    sampType='GLL' #how to generate nodes
                        #'random', 'uniform', 'GL' (Gauss-Legendre), 'Clenshaw',
    fType='type1'     # Type of model function used as simulator
    #----------------------------------
    # Create nNodes random nodal points over qBound range and function value at the nodes
    samps_=sampling.trainSample(sampleType=sampType,qInfo=qBound,nSamp=nNodes)
    qNodes=samps_.q
    fNodes=analyticTestFuncs.fEx1D(qNodes,fType,qBound).val

    # Generate test points
    qTestFull=np.linspace(qBound[0],qBound[1],nTest)
    qTest=np.linspace(min(qNodes),max(qNodes),nTest)

    # Use nodal values in Lagrange interpolation to predict at test points
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
    # Settings of the discrete samples in space of param1 & param2
    nNodes=[5,4]   #number of nodes in space of parameters q1, q2
    sampType=['GLL',      #Method to draw samples for q1, q2
              'unifSpaced']
    qBound=[[-0.75,1.5],  #param_k-space <range_k
            [-0.5 ,2.5]]

    # Settings of the exact response surface
    domRange=[[-2,2], #range in the k-th direction over which the analytical response is evaluated
              [-3,3]]
    # Test points over range1,2 and qBound1,2
    nTest=[100,101] #number of test points for plot the exact response
    #-----------------------------------------------------------------------------
    p=len(nNodes)
    # (1) Create nodal sets over the parameter space (each node=one joint sample)
    # Generate Gauss-Legendre points over qBounds[0] and qBounds[1]
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)
    # Response at the GL samples
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd').val

    # (3) Use response values at the sampled nodes and predict at test points.The test points are generated over q1Boundxq2Bound. These points make a uniform mesh that is used for contourplot
    qTestList=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTestList.append(qTest_.q)
    # Use nodal values in Lagrange interpolation to predict at test points
    fTest=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTestList,liDict={'testRule':'tensorProd'}).val

    # (4) Evaluate the exact response over a 2D mesh which covers the whole space range1xrange2 (exact response surface)
    qTestFull=[]
    for i in range(p):
        qTestFull_=np.linspace(domRange[i][0],domRange[i][1],nTest[i])  #test points in param1 space
        qTestFull.append(qTestFull_)

    fTestFull=analyticTestFuncs.fEx2D(qTestFull[0],qTestFull[1],'type1','tensorProd').val   #response value at the test points
    fTestFullGrid=fTestFull.reshape((nTest[0],nTest[1]),order='F').T
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F').T

    #(5) 2D Contour Plots
    plt.figure(figsize=(16,8));
    plt.subplot(1,2,1)
    ax=plt.gca()
    CS1 = plt.contour(qTestFull[0],qTestFull[1],fTestFullGrid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    qNodesGrid=reshaper.vecs2grid(qNodes)  #2d mesh
    plt.plot(qNodesGrid[:,0],qNodesGrid[:,1],'o',color='r',markersize=6)
    plt.xlabel(r'$q_1$',fontsize=20);plt.ylabel(r'$q_2$',fontsize=20);
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('Exact Response Surface')
    plt.subplot(1,2,2)
    ax=plt.gca()
    CS2 = plt.contour(qTestList[0],qTestList[1],fTestGrid,20)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qNodesGrid[:,0],qNodesGrid[:,1],'o',color='r',markersize=6)
    plt.xlabel('q1',fontsize=20);plt.ylabel('q2',fontsize=20);
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('Response Surface by Lagrange Interpolation from Sampled Values')
    plt.xlim(domRange[0])
    plt.ylim(domRange[1])
    plt.show()
#
def lagInt_3d_test():
    """
    Test Lagrange inerpolation over a 3D parameter space.
    """
    #----- SETTINGS -------------------
    # Settings of the discrete samples in space of param1 & param2
    nNodes=[8,7,6] #number of Gauss quadratures in each of the 3 parameter spaces
    sampType=['GLL',      #Method to draw samples for q1, q2
              'unifSpaced',
              'Clenshaw']
    qBound=[[-0.75,1.5], #1D parameter spaces
            [-0.5 ,2.5],
            [1,3]]
    # Test points over each 1D parameter space
    nTest=[10,11,12]
    #parameters in Ishigami function
    a=7
    b=0.1
    #----------------------------------
    p=len(nNodes)
    # (1) Generate uniformly-spaced samples over each of the 1D parameter spaces
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)

    # (2) Function values at the samples
    fNodes=analyticTestFuncs.fEx3D(qNodes[0],qNodes[1],qNodes[2],'Ishigami','tensorProd',{'a':a,'b':b}).val

    # (3) Create the test points
    qTest=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTest.append(qTest_.q)
    # (4) Compute exact and predicted-by-Lagrange inerpolation values of function
    # Exact Value
    fTestEx=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',{'a':a,'b':b}).val
    # (5) Construct Lagrange interpolation from the nodal values and make predictions at test points
    fInterp=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTest,liDict={'testRule':'tensorProd'}).val

    # (6) Plot
    plt.figure(figsize=(10,7))
    plt.subplot(2,1,1)
    fInterp_=fInterp.reshape(np.asarray(np.prod(np.asarray(nTest))),order='F')
    plt.plot(fInterp_,'-ob',mfc='none',label='Lagrange Interpolation from Nodal Values')
    plt.plot(fTestEx,'--xr',ms=5,label='Analytical Value')
    plt.ylabel(r'$f(q1,q2,q3)$',fontsize=18)
    plt.xlabel(r'Test Parameters',fontsize=18)
    plt.legend(loc='best',fontsize=18)
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(abs(fInterp_-fTestEx)/(fTestEx)*100,'-sk')
    plt.ylabel(r'$|f_{interp}(q)-f_{Analytic}(q)|/f_{Analytic}(q) \%$',fontsize=15)
    plt.xlabel(r'Test Parameters',fontsize=18)
    plt.grid(alpha=0.4)
    plt.show()
#
def lagInt_Quads2Line_test():
    """
    Test lagInt_Quads2Line().
    A set of Gauss-Legendre points in a 2D parameters plane along with associated response are given. The aim is to construct a Lagrange interpolation based on these sets and interpolate the response at all points over a straight line relying in the 2D parameter plane.
    """
    #----- SETTINGS --------------------------------------------------------------
    # Settings of the discrete samples in space of param1 & param2
    nNodes=[9,9]   #number of (non-uniform=Gauss-Legendre) nodes in 1d parameter spaces
    sampType=['GLL',      #Method to draw samples for q1, q2
              'unifSpaced']
    qBound=[[-0.75,1.5],  #param_k-space <range_k
            [-0.8 ,2.5]]  #(line k: range for param k)
    # Define the line in qBound[0]xqBound[1] plane over which interpolation is to be done
    lineDef={'start':[1.4,2.3],    #coordinates of the line's starting point in the q1-q2 plane
             'end':[-0.7,-0.2],    #coordinates of the line's end point in the q1-q2 plane
             'noPtsLine':100
            }
    #-----------------------------------------------------------------------------
    p=len(nNodes)
    # (1) Create nodal sets over the parameter space (each node=one joint sample)
    # Generate Gauss-Legendre points over qBounds[0] and qBounds[1]
    qNodes=[]
    for i in range(p):
        qNodes_=sampling.trainSample(sampleType=sampType[i],qInfo=qBound[i],nSamp=nNodes[i])
        qNodes.append(qNodes_.q)
    # Response at the GL samples
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd').val

    #(2) Interpolate from the nodal set to the points over the line defined above
    qLine,fLine=lagInt_Quads2Line(fNodes,qNodes,lineDef)

    #(3) Plot results
    plt.figure(figsize=(8,5))
    plt.plot(qLine[0],fLine,'-ob',label='Lagrange Interpolation')
    # exact response
    fLine_ex=analyticTestFuncs.fEx2D(qLine[0],qLine[1],'type1','comp').val
    plt.plot(qLine[0],fLine_ex,'-xr',label='Exact')
    plt.title('%d x%d interpolating nodes in Q1xQ2 space.' %(nNodes[0],nNodes[1]))
    plt.xlabel('q1');
    plt.ylabel('Response')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()
