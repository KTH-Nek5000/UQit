###########################################################
#     Lagrange Interpolation based on tensor-product
# Using response values at a given nodal set and interpolate
#     at some test points within a space defined by the nodes.
# NOTE: To avoid Runge's phenomenon, the nodal set should 
#       be non-uniformly distributed in each dimension of 
#       parameter space. A good example of proper nodal set
#       are Gauss quadratures.  
###########################################################
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
import analyticTestFuncs
import sampling
import pce
import reshaper
#
def lagrangeBasis_singleVar(qNodes,k,Q):
    """
      Construct Lagrange bases L_k(q) using n nodes qNode and evaluates the bases at m test points Q.
      Note that k=0,1,...,n-1. 
      where k=1,2,...,n-1

      Parameters
      ----------
         qNodes: single variate training nodal set, numpy 1d array: (n)
         Q: a vector of m test samples 
         k: k-th basis: L_k(q) are constructed from the nodal set and are polymoials of order (n-1). 
         L_k(Q)=(Q-q_0)(Q-q_1)...(Q-q_{k-1})(Q-q_{k+1})(Q-q_{n-1})/(q_k-q_0)(q_k-q_1)...(q_k-q_{k-1})(q_k-q_{k+1})(q_k-q_{n-1})
         => L_k(Q)=\Prod_{k=0,k!=j}^{n-1} (Q-qj)/(qk-qj)
      Returns
      -------
      prod: n-by-m numpy array
          Values of Lk(Q) for k=0,1,...,m 
    """
    n=qNodes.size
    m=Q.shape[-1]
    prod=np.zeros((k.shape[-1],m))
    for k_ in k:
        prod_=1.0
        for j in range(n):
            if j!=k_:
               prod_*=(Q-qNodes[j])/(qNodes[k_]-qNodes[j])
        prod[k_,:]=prod_       
    return prod
#
def lagrangeInterpol_singleVar(fNodes,qNodes,Q):
    """
       Lagrange interpolation of order (n-1) constructed from n nodes qNodes (nodal set for single variate q) and is evaluated at Q\in[min(qNodes),max(qNodes)]. 
          qNodes: training nodes, single variate numpy vectors of length n
          fNodes: response at qNodes, single-response vectors of length n 
          Q: a vector of m test samples           
          F(Q)=sum_{k=1}^n fNodes(k) L_k(Q)
          Lagrange Bases L_k(q) are constructed from the nodal set and are polymoials of order (n-1). L_k(Q)=(Q-q_1)(Q-q_2)...(Q-q_{k-1})(Q-q_{k+1})(Q-q_n)/(q_k-q_1)(q_k-q_2)...(q_k-q_{k-1})(q_k-q_{k+1})(q_k-q_n)
          fNodes are the values of a true unobserved function f at nodal sets qNodes.
          qNodes and Q are in the same space; make sure Q\in[min(qNodes),max(qNodes)]
    """  
    Q = np.array(Q, copy=False, ndmin=1)
    if (np.all(Q>max(qNodes))):
      print('ERORR in lagrangeInterpol_singleVar: Q>max(qNodes)')
    if (np.all(Q<min(qNodes))):
      print('ERORR in lagrangeInterpol_singleVar: Q<min(qNodes)')
    if (fNodes.size!=qNodes.size):
       print('ERROR in lagrangeInterpol: qNodes and fNodes should be of the same size.')
    n=fNodes.size
    k=np.arange(n)
    Lk=lagrangeBasis_singleVar(qNodes,k,Q)
    fInterp=np.matmul(fNodes[None,:],Lk).T
    return fInterp
#
def lagrangeInterpol_multiVar(fNodes,qNodes,Q,liDict):
    """
       Lagrange interpolation of order (n_1-1)*(n_2-1)*...*(n_p-1) is constructed in a p-dimensional parameter space. Note, n_k refers to number of nodes in the i-th direction of the parameter space.

       Parameters
       ----------
         qNodes: training nodes, a list of [qNodes_1|qNodes_2|...|qNodes_p], where qNodes_k is a 1D numpy array of n_k nodes in the k-th parameter dimension. 
         fNodes: response value at training nodes, p-D numpy array of shape (n_1,n_2,...,n_p) or a 1D array of length n_1*n_2*...*n_p
         Q: a list of size p containing test samples at p-dimensions.
            Q=[Q_0,Q_1,...,Q_{p-1}] where Qi= 1d numpy array of size mi, i=0,1,...,p-1
         liDict: A dictionary to set different options for Lagrange interpolation over the pD space
         keys:
         'testRule': The rule for treating the multi-dim test points
               'tensorProd': a tensor product grid is constructed from Q_i in list Q
                             total number of sample points m=m0*m1*m_{p-1}
               'set': All m_i for i=0,1,...p-1 should be equal. A set of m=m_i test points is considered.

         F(Q)=sum_{k1=1}^n_1 ... sum_{kp=1}^n_p [fNodes(k1,k2,...,kp) L_k1(Q[:,1]) L_k2(Q[:,2]) ... L_kp(Q[:,p])]
              where, L_ki(Qp) is the single-variate lagrange interpolant in i-th direction. 
              fNodes are the values of a true unobserved function f at qNodes.
              qNodes and Q can be in any space, but make sure Q[:,i]\in[min(qNodes_i),max(qNodes_i)] for i=1,2,...,p
    """
##    Q = np.array(Q, copy=False, ndmin=1)
    Q=np.asarray(Q)
    p=len(qNodes)  #number of dimensions of parameter space
    nList=[] #list of the number of nodes in each dimension
    mList=[] #list of the number of test points
    for i in range(p):
        n_=qNodes[i].shape[0]
        nList.append(n_)
        mList.append(Q[i].shape[0])
    if fNodes.ndim==1:
       # NOTE: the smaller index changes slowest (fortran like)
       fNodes=np.reshape(fNodes,nList,order='F')
    #check the arguments 
    testRule=liDict['testRule']
    if testRule!='tensorProd' and testRule!='set':
       raise ValueError("ERROR in lagrangeInterpol_multiVar: Invalid value for 'tensorProd'") 
    for i in range(p):
        if (np.all(Q[i]>np.amax(qNodes[i],axis=0))):
           print('ERORR in lagrangeInterpol_multiVar: Q>max(qNodes)')
        if (np.all(Q[i]<np.amin(qNodes[i],axis=0))):
           print('ERORR in lagrangeInterpol_multiVar: Q<min(qNodes)')
        if (fNodes.shape[i]!=nList[i]):
           print('ERROR in lagrangeInterpol_multiVar: qNodes and fNodes should be of the same size/shape.')
    if (p!=len(Q)):
          print('ERROR in lagrangeInterpol_multiVar: qNodes and sampled Q should be of the same dimension.')
    #Construct and evaluate Lagrange interpolation
    idxTest=[]    #List of indices counting the test points
    Lk=[]         #List of Lagrange bases
    for i in range(p):
        idxTest.append(np.arange(mList[i]))
        k=np.arange(nList[i])
        Lk_=lagrangeBasis_singleVar(qNodes[i],k,Q[i]) #Basis at the i-th dim
        Lk.append(Lk_)

    if testRule=='tensorProd':
       idxTestGrid=reshaper.vecs2grid(idxTest)    
    elif testRule=='set':
       idxTestGrid=idxTest[0]  #same for all dimensions

    mTot=idxTestGrid.shape[0]   #total number of test points
    fInterp=np.zeros(mTot)
    if p>2:
       mulInd=[[i for i in range(p-1,-1,-1)]]*(p-1)   #list of indices for tensor-dot
    else:   
       mulInd=p 

    for j in range(mTot):  #test points
        idxTest_=idxTestGrid[j]
        if testRule=='tensorProd':
           Lk_prod=Lk[0][:,int(idxTest_[0])]
           for i in range(1,p):
               Lk_prod=np.tensordot(Lk_prod,Lk[i][:,int(idxTest_[i])],0)
        elif testRule=='set':
           Lk_prod=Lk[0][:,int(idxTest_)]
           for i in range(1,p):
               Lk_prod=np.tensordot(Lk_prod,Lk[i][:,int(idxTest_)],0)
        fInterp[j]=np.tensordot(Lk_prod,fNodes,mulInd)
    if testRule=='tensorProd':
       fInterp=fInterp.reshape(mList,order='F')
    return fInterp
#
def lagrangeInterpol_Quads2Line(fQuads,quads,lineDef):
    """ 
       Lagrange interpolation of the response from tensor-product quadratures (e.g. Gauss-Legendre points) in a 2D plane of parameters to the points located on a straight line lying in the same parameter plane. 
       quads: [Q1|Q2]  list of quadratures in directions 1 and 2. Qi is a 1d numpy array of length ni
       fQuads: Values of the response at the quadratures: 1D numpy array of length n1*n2
       lineDef: a dictionary defining the line 
                lineDef={'lineStart'=[q1Start,q2Start],    #line's starting point
                         'lineEnd=[q1End,q2End]',          #line's end point 
                         'noPtsLine'=<inetger>}                #number interpolation points on the line
    """
    p=len(quads)
    nQ=[quads[0].shape[0],quads[1].shape[0]]
    if (fQuads.ndim==1):
       fQuads=np.reshape(fQuads,nQ,'F')  
    #limits of the parameters space
    qBound=[[min(quads[0]),max(quads[0])],
            [min(quads[1]),max(quads[1])]]
    #(1) Check if the line is relying on the parameters plane
    lineStart=lineDef['start'] 
    lineEnd  =lineDef['end'] 
    for i in range(p):
       if (lineStart[i]<qBound[i][0] or lineStart[i]>qBound[i][1]):
          print('ERROR in lagrangeInterpol_Quads2Line: line cannot be outside of the parameters plane. Issue in lineStart in parameter direction %d' %i)
       if (lineEnd[i]<qBound[i][0] or lineEnd[i]>qBound[i][1]):
          print('ERROR in lagrangeInterpol_Quads2Line: line cannot be outside of the parameters plane. Issue in lineEnd in parameter direction %d' %i)
          
    #(2) Generate interpolation points over the line
    noPtsLine=lineDef['noPtsLine']
    q1=np.linspace(lineStart[0],lineEnd[0],noPtsLine)
    slope=(lineEnd[1]-lineStart[1])/(lineEnd[0]-lineStart[0])
    q2=slope*(q1-lineEnd[0])+lineEnd[1]
    
    #(3) Lagerange interpolation from quadratures to the points on the line
    qLine=[q1,q2]
    fLine=lagrangeInterpol_multiVar(fQuads,quads,qLine,{'testRule':'set'})
    return qLine,fLine
#
#
# Tests
#
def lagrangeInterpol_singleVar_test():
    """
       Test for lagrangeInterpol_singleVar(.....)
       Take nNodes random samples from the 1D parameter space at which the value of function f is known. Use these in Lagrange interpolation to predict values of f at all q in the subset of the parameter space defined by min and max of the samples. 
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
    fNodes=analyticTestFuncs.fEx1D(qNodes,fType)

    # Generate test points
    qTestFull=np.linspace(qBound[0],qBound[1],nTest)
    qTest=np.linspace(min(qNodes),max(qNodes),nTest)

    # Use nodal values in Lagrange interpolation to predict at test points
    fInterpTest=lagrangeInterpol_singleVar(fNodes,qNodes,qTest)

    # Plot
    fTestFull=analyticTestFuncs.fEx1D(qTestFull,fType)
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
#//////////////////////////////////////
def lagrangeInterpol_multiVar_test2d():
    """
       Test for lagrangeInterpol_multiVar(.....) - 2D parameter space - Tensor Product
       Left Fig: Contourlines of the exact response f evaluated at 2d space range1xrange2.
       Right Fig: Take (nNodes1,nNodes2) random samples from a 2D parameter space q1Boundxq2Bound that is a subset of range1xrange2.Use these nodal values in a Lagrange interpolation to predict values of f at all q \in q1Boundxq2Bound.
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
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd')

    # (3) Use response values at the sampled nodes and predict at test points.The test points are generated over q1Boundxq2Bound. These points make a uniform mesh that is used for contourplot
    qTestList=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTestList.append(qTest_.q)
    # Use nodal values in Lagrange interpolation to predict at test points
    fTest=lagrangeInterpol_multiVar(fNodes,qNodes,qTestList,{'testRule':'tensorProd'})    

    # (4) Evaluate the exact response over a 2D mesh which covers the whole space range1xrange2 (exact response surface)
    qTestFull=[]
    for i in range(p):
        qTestFull_=np.linspace(domRange[i][0],domRange[i][1],nTest[i])  #test points in param1 space
        qTestFull.append(qTestFull_)

    fTestFull=analyticTestFuncs.fEx2D(qTestFull[0],qTestFull[1],'type1','tensorProd')   #response value at the test points
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
def lagrangeInterpol_multiVar_test3d():
    """
       Test for lagrangeInterpol_multiVar(.....) - 3D parameter space - Tensor Product
        - A 3d parameter space Q=Q1xQ2xQ3 is given (tensor product).
        - We take samples n1xn2xn3 over the space and evaluate Ishigami function at each sample (node). 
        - Use Lagrange interpolation with the above sets of nodes to predict function values all over Q. 
        - Compare the predicted function values with the analytical (Ishigami) values. 
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
    fNodes=analyticTestFuncs.fEx3D(qNodes[0],qNodes[1],qNodes[2],'Ishigami','tensorProd',{'a':a,'b':b})
   
    # (3) Create the test points
    qTest=[]
    for i in range(p):
        qTest_=sampling.testSample(sampleType='unifSpaced',qBound=qBound[i],nSamp=nTest[i])
        qTest.append(qTest_.q)
    # (4) Compute exact and predicted-by-Lagrange inerpolation values of function
    # Exact Value 
    fTestEx=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',{'a':a,'b':b})
    # (5) Construct Lagrange interpolation from the nodal values and make predictions at test points
    fInterp=lagrangeInterpol_multiVar(fNodes,qNodes,qTest,{'testRule':'tensorProd'})
    
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
def lagrangeInterpol_Quads2Line_test():
    """
       Test lagrangeInterpol_Quads2Line_test(.....):
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
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd')

    #(2) Interpolate from the nodal set to the points over the line defined above
    qLine,fLine=lagrangeInterpol_Quads2Line(fNodes,qNodes,lineDef) 

    #(3) Plot results
    plt.figure(figsize=(8,5))
    plt.plot(qLine[0],fLine,'-ob',label='Lagrange Interpolation')
    # exact response
    fLine_ex=analyticTestFuncs.fEx2D(qLine[0],qLine[1],'type1','pair')
    plt.plot(qLine[0],fLine_ex,'-xr',label='Exact')
    plt.title('%d x%d interpolating nodes in Q1xQ2 space.' %(nNodes[0],nNodes[1]))
    plt.xlabel('q1');
    plt.ylabel('Response')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.show()    
