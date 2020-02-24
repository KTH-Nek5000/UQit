###########################################################
#           Lagrange Interpolation
# Using response values at a given nodal set and interpolate
#     at test points within a space defined by the nodes.
# NOTE: To avoid Runge's phenomenon, the nodal set should 
#       be non-uniformly distributed in each dimension of 
#       parameter space. A good example of proper nodal set
#       are Gauss quadratures.  
###########################################################
# Saleh Rezaeiravesh, salehr@kth.se
#----------------------------------------------------------
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append('../analyticFuncs/')
sys.path.append('../plot/')
sys.path.append('../gPCE/')
sys.path.append('../general/')
sys.path.append('../nodes/')
import analyticTestFuncs
import plot2d
import gpce
import reshaper
import nodes

#////////////////////////////////////////
def lagrangeBasis_singleVar(qNodes,k,Q):
    """
      Construct Lagrange bases L_k(q) based on n qNodes and return L_k(Q), note k=1,2,...,n-1
         qNodes: single variate training nodal set, numpy 1d array: (n)
         Q: a vector of m test samples 
         k: k-th basis: L_k(q) are constructed from the nodal set and are polymoials of order (n-1). 
         L_k(Q)=(Q-q_0)(Q-q_1)...(Q-q_{k-1})(Q-q_{k+1})(Q-q_{n-1})/(q_k-q_0)(q_k-q_1)...(q_k-q_{k-1})(q_k-q_{k+1})(q_k-q_{n-1})
         => L_k(Q)=\Prod_{k=0,k!=j}^{n-1} (Q-qj)/(qk-qj)
    """
    n=qNodes.size
    prod=1.0
    for j in range(n):
        if j!=k:
           prod*=(Q-qNodes[j])/(qNodes[k]-qNodes[j])
    return prod

#/////////////////////////////////////////
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
    #check1
    if (np.all(Q>max(qNodes))):
      print('ERORR in lagrangeInterpol_singleVar: Q>max(qNodes)')
    if (np.all(Q<min(qNodes))):
      print('ERORR in lagrangeInterpol_singleVar: Q<min(qNodes)')
    #check2
    if (fNodes.size!=qNodes.size):
       print('ERROR in lagrangeInterpol: qNodes and fNodes should be of the same size.')
    n=fNodes.size
    fInterp=0.0;
    for k in range(n):
        L_k=lagrangeBasis_singleVar(qNodes,k,Q)
        fInterp+=fNodes[k]*L_k
    return fInterp

#/////////////////////////////////////////
def lagrangeInterpol_multiVar(fNodes,qNodes,Q,method):
    """
       method='tensorProd'
       Lagrange interpolation of order (n_1-1)*(n_2-1)*...*(n_p-1) is constructed in a p-dimensional parameter space. Note, n_k refers to number of nodes in the i-th direction of the parameter space.
          if method=='tensorProd', then
              qNodes: training nodes, a list of [qNodes_1|qNodes_2|...|qNodes_p], where qNodes_k is a 1D numpy array of n_k nodes in the k-th parameter dimension. 
              fNodes: response value at training nodes, p-D numpy array of shape (n_1,n_2,...,n_p) or a 1D array of length n_1*n_2*...*n_p
              Q: numpy array of m test samples and p-dimensions (m x p)
                 F(Q)=sum_{k1=1}^n_1 ... sum_{kp=1}^n_p [fNodes(k1,k2,...,kp) L_k1(Q[:,1]) L_k2(Q[:,2]) ... L_kp(Q[:,p])]
                 where, L_ki(Qp) is the single-variate lagrange interpolant in i-th direction. 
              fNodes are the values of a true unobserved function f at qNodes.
              qNodes and Q can be in any space, but make sure Q[:,i]\in[min(qNodes_i),max(qNodes_i)] for i=1,2,...,p
    """
    if method=='tensorProd': #Tensor product in multi-dimensional parameter space  
       Q = np.array(Q, copy=False, ndmin=1)
       p=len(qNodes)  #number of dimensions of parameter space
       n=[]; #list of the number of nodes in each dimension
       for i in range(p):
           n_=qNodes[i].shape[0]
           n.append(n_)
       m=Q.shape[0]        #number of samples for each of d-parameters

       #Reshape fNodes from 1d array of size (n1*n2*...*np) to p-d array of (n1,n2,...,np)
       # NOTE: the loop of later parameter is assumed to be the outermost
       if fNodes.ndim==1:
          if p==2:
             fNodes=np.reshape(fNodes,(n[0],n[1]),'F')
          elif p==3:
             fNodes=np.reshape(fNodes,(n[0],n[1],n[2]),'F')
          else:
             print('ERROR in lagrangeInterpol_multiVar: parameter dimensions >3 are not currently supported! p=%d' %p) 
 
       #check1
       for i in range(p):
           if (np.all(Q[:,i]>np.amax(qNodes[i][:],axis=0))):
              print('ERORR in lagrangeInterpol_multiVar: Q>max(qNodes)')
           if (np.all(Q[:,i]<np.amin(qNodes[i][:],axis=0))):
              print('ERORR in lagrangeInterpol_multiVar: Q<min(qNodes)')
           #check2
           if (fNodes.shape[i]!=n[i]):
              print('ERROR in lagrangeInterpol_multiVar: qNodes and fNodes should be of the same size/shape.')
       if (p!=Q.shape[1]):
          print('ERROR in lagrangeInterpol_multiVar: qNodes and sampled Q should be of the same dimension.')

       fInterp=np.zeros(m);  #interpolated value at each of the m test samples       
       if (p==2): #2D parameter space
          for i in range(m):
              for k1 in range(n[0]):
                  L_k1=lagrangeBasis_singleVar(qNodes[0][:],k1,Q[i,0])
                  for k2 in range(n[1]):                 
                      L_k2=lagrangeBasis_singleVar(qNodes[1][:],k2,Q[i,1])
                      fInterp[i]+=fNodes[k1,k2]*L_k1*L_k2
       elif (p==3): #3D parameter space
          for i in range(m):
              for k1 in range(n[0]):
                  L_k1=lagrangeBasis_singleVar(qNodes[0][:],k1,Q[i,0])
                  for k2 in range(n[1]):                 
                      L_k2=lagrangeBasis_singleVar(qNodes[1][:],k2,Q[i,1])
                      for k3 in range(n[2]):                 
                          L_k3=lagrangeBasis_singleVar(qNodes[2][:],k3,Q[i,2])
                          fInterp[i]+=fNodes[k1,k2,k3]*L_k1*L_k2*L_k3
       else:
          print('ERROR in lagrangeInterpol_multiVar: p should be less than 4!') 
                     
    else: 
       print('ERROR in lagrangeInterpol_multiVar: for multi-dimensions, only tensor product is valid')
    return fInterp

#////////////////////////////////////////
def lagrangeBasis_multiVar_metric(qNodes,k,Q):
    """
      REPRIEVED: the tensor-product is used instead, see lagrangeInterpol_multiVar(...)
      Construct L_k(q) based on n qNodes and return L_k(Q)
         qNodes: multi(d)-variate nodal set, numpy array (n,d)
         Q: a vector of m samples and d-dimension: (m,d) 
         prod: numpy vector of length m
         k: k-th basis: L_k(q) are constructed from the nodal set and are polymoials of order (n-1). 
         L_k(Q)=(Q-q_0)(Q-q_1)...(Q-q_{k-1})(Q-q_{k+1})(Q-q_{n-1})/(q_k-q_0)(q_k-q_1)...(q_k-q_{k-1})(q_k-q_{k+1})(q_k-q_{n-1})
         => L_k(Q)=\Prod_{k=0,k!=j}^{n-1} (Q-qj)/(qk-qj)
    """
    n=qNodes.shape[0]   #number of nodes 
    d=qNodes.shape[1]   #number of dimensions of parameter space
    prod=1.0
    for j in range(n):
        if j!=k:
           sDen=np.linalg.norm((qNodes[k,:]-qNodes[j,:]))
           sNum=np.linalg.norm((Q-qNodes[j,:]),axis=1)
           prod*=sNum/sDen
    return prod

#///////////////////////////////
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
    nQ1=len(quads[0])
    nQ2=len(quads[1])
    if (fQuads.ndim==1):
       fQuads=np.reshape(fQuads,(nQ1,nQ2),'F')  
    #limits of the parameters space
    qBound=[[min(quads[0]),max(quads[0])],
            [min(quads[1]),max(quads[1])]]
    #(1) Check if the line is relying on the parameters plane
    lineStart=lineDef['start'] 
    lineEnd  =lineDef['end'] 
    for i in range(2):
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
    qLine=reshaper.vecsGlue(q1,q2)
    fLine=lagrangeInterpol_multiVar(fQuads,quads,qLine,'tensorProd')
    return qLine,fLine
    

#############################
# External Functions: TESTS
#############################
def lagrangeInterpol_singleVar_test():
    """
       Test for lagrangeInterpol_singleVar(.....)
       Take nNodes random samples from the 1D parameter space at which the value of function f is known. Use these in Lagrange interpolation to predict values of f at all q in the subset of the parameter space defined by min and max of the samples. 
    """
    #----- SETTINGS -------------------
    nNodes=15     #number of nodes
    qBound=[-1,3] #range over which nodes are randomly chosen
    nTest=100     #number of test points for plot
    how='Clenshaw'  #how to generate nodes
              #'random', 'uniform', 'GL' (Gauss-Legendre), 'Clenshaw', 
    #---------------------------------- 
    # create nNodes random nodal points over qBound range and function value at the nodes
    qNodes=np.zeros(nNodes)
    fNodes=np.zeros(nNodes)
    if how=='random': 
       xi=np.random.uniform(0,1,size=[nNodes])
    elif how =='uniform':
       xi=np.linspace(0,1,nNodes)
    elif how =='GL':
       xi,wXI=gpce.GaussLeg_ptswts(nNodes)  #on [-1,1]
    elif how=='Clenshaw':
       xi=nodes.Clenshaw_pts(nNodes)
 
    xi=gpce.mapFromUnit(xi,[0,1])    #map to [0,1]
    qNodes=(qBound[1]-qBound[0])*xi+qBound[0]
    fNodes=analyticTestFuncs.fEx1D(qNodes)

    #test points
    qTestFull=np.linspace(qBound[0],qBound[1],nTest)
    qTest=np.linspace(min(qNodes),max(qNodes),nTest)

    # Use nodal values in Lagrange interpolation to predict at test points
    fInterpTest=lagrangeInterpol_singleVar(fNodes,qNodes,qTest)

    #plot
    fTestFull=analyticTestFuncs.fEx1D(qTestFull)
    plt.figure(figsize=(12,7))
    plt.plot(qTestFull,fTestFull,'--r',lw=2,label='Exact f(q)')
    plt.plot(qTest,fInterpTest,'-b',lw=2,label='Interpolated f(q) by Lagrange Bases')
    plt.plot(qNodes,fNodes,'oc',markersize='8',label='Nodes')
    plt.legend(loc='best',fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid()
    plt.xlabel(r'$q$',fontsize=26)
    plt.ylabel(r'$f(q)$',fontsize=26)
    print('Rerun to change distribution of nodal points!')
#    plt.title('Nodes are chosen randomly over [%g,%g]' %(qBound[0],qBound[1]))
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1000/float(DPI),500/float(DPI))
    plt.savefig('../testFigs/lag1d4'+'.png',bbox_inches='tight')
    plt.show()

#//////////////////////////////////////
def lagrangeInterpol_multiVar_test2d():
    """
       Test for lagrangeInterpol_multiVar(.....) - 2D parameter space - Tensor Product
       Left Fig: Contourlines of the exact response f evaluated at 2d space range1xrange2.
       Right Fig: Take (nNodes1,nNodes2) random samples from a 2D parameter space q1Boundxq2Bound that is a subset of range1xrange2.Use these nodal values in a Lagrange interpolation to predict values of f at all q \in q1Boundxq2Bound.
    """
    #----- SETTINGS --------------------------------------------------------------
    # Settings of the discrete samples in space of param1 & param2
    nNodes=[5,5]   #number of (non-uniform=Gauss-Legendre) nodes in 1d parameter spaces
    qBound=[[-0.75,1.5],  #param_k-space <range_k
            [-0.5 ,2.5]]  
    # Settings of the exact response surface
    domRange=[[-2,2], #range in the k-th direction over which the analytical response is evaluated
           [-3,3]]
    # Test points over range1,2 and qBound1,2
    nTest=[100,100] #number of test points for plot the exact response
    #-----------------------------------------------------------------------------
    # (1) Create nodal sets over the parameter space (each node=one joint sample)    
    # Generate Gauss-Legendre points over qBounds[0] and qBounds[1]
    qNodes=[]
    for i in range(2):
        xi,wXI=gpce.GaussLeg_ptswts(nNodes[i])
        qNodes_=gpce.mapFromUnit(xi,qBound[i]) 
        qNodes.append(qNodes_)
    # Response at the GL samples
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd')

    # (3) Use response values at the sampled nodes and predict at test points.The test points are generated over q1Boundxq2Bound. These points make a uniform mesh that is used for contourplot
    qTestList=[]
    for i in range(2):
        qTest_ =np.linspace(qBound[i][0],qBound[i][1],nTest[i])  #test points in param_i space
        qTestList.append(qTest_)
    # Use nodal values in Lagrange interpolation to predict at test points
    qTest=reshaper.vecs2grid(qTestList[0],qTestList[1]) 
    fTest=lagrangeInterpol_multiVar(fNodes,qNodes,qTest,'tensorProd')    
    fTestGrid=fTest.reshape((nTest[0],nTest[1]),order='F').T

    # (4) Evaluate the exact response over a 2D mesh which covers the whole space range1xrange2 (exact response surface)
    qTestFull=[]
    for i in range(2):
        qTestFull_=np.linspace(domRange[i][0],domRange[i][1],nTest[i])  #test points in param1 space
        qTestFull.append(qTestFull_)

    fTestFull=analyticTestFuncs.fEx2D(qTestFull[0],qTestFull[1],'type1','tensorProd')   #response value at the test points
    fTestFullGrid=fTestFull.reshape((nTest[0],nTest[1]),order='F').T

    #(5) 2D Contour Plots
    plt.figure(figsize=(16,8));
    plt.subplot(1,2,1)
    ax=plt.gca()
    CS1 = plt.contour(qTestFull[0],qTestFull[1],fTestFullGrid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=15,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    qNodesGrid=reshaper.vecs2grid(qNodes[0],qNodes[1])  #2d mesh 
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

#///////////////////////////////////////
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
    nNodes=[7,8,7] #number of Gauss quadratures in each of the 3 parameter spaces
    qBound=[[-0.75,1.5], #1D parameter spaces
            [-0.5 ,2.5],
            [1,3]]
    # Test points over each 1D parameter space
    nTest=[8,6,7]
    #parameters in Ishigami function
    a=0.2
    b=0.1
    #---------------------------------- 
    # (1) Generate uniformly-spaced samples over each of the 1D parameter spaces
    qNodes=[]
    for i in range(3):
        qNodes_=np.linspace(qBound[i][0],qBound[i][1],nNodes[i])
        qNodes.append(qNodes_)

    # (2) Function values at the samples
    fNodes=analyticTestFuncs.fEx3D(qNodes[0],qNodes[1],qNodes[2],'Ishigami','tensorProd',{'a':a,'b':b})
   
    # (3) Create the test points
    qTest=[]
    for i in range(3):
        qTest_=np.linspace(qBound[i][0],qBound[i][1],nTest[i])
        qTest.append(qTest_)
    # (4) Compute exact and predicted-by-Lagrange inerpolation values of function
    # Exact Value 
    fTestEx=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',{'a':a,'b':b})
    # (5) Construct Lagrange interpolation from the nodal values and make predictions at test points
    qTestGrid=reshaper.vecs2grid3d(qTest[0],qTest[1],qTest[2])
    fInterp=lagrangeInterpol_multiVar(fNodes,qNodes,qTestGrid,'tensorProd')
    
    # (6) Plot
    plt.figure(figsize=(10,7))
    plt.subplot(2,1,1)
    plt.plot(fInterp,'-ob',label='Lagrange Interpolation from Nodal Values')
    plt.plot(fTestEx,'--xr',label='Analytical Value')
    plt.ylabel(r'$f(q1,q2,q3)$')
    plt.xlabel(r'Nodal set')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(abs(fInterp-fTestEx)/fTestEx,'-sk')
    plt.ylabel(r'$|f_{interp}(q)-f_{Analytic}(q)|/f_{Analytic}(q)$')
    plt.xlabel(r'Nodal set')
    plt.grid()
    plt.show()

#/////////////////////////////////////////////////////////
def lagrangeInterpol_Quads2Line_test():
    """
       Test lagrangeInterpol_Quads2Line_test(.....):
       A set of Gauss-Legendre points in a 2D parameters plane along with associated response are given. The aim is to construct a Lagrange interpolation based on these sets and interpolate the response at all points over a straight line relying in the 2D parameter plane.    
    """
    #----- SETTINGS --------------------------------------------------------------
    # Settings of the discrete samples in space of param1 & param2
    nNodes=[9,9]   #number of (non-uniform=Gauss-Legendre) nodes in 1d parameter spaces
    qBound=[[-0.75,1.5],  #param_k-space <range_k
            [-0.8 ,2.5]]  #(line k: range for param k)
    # Define the line in qBound[0]xqBound[1] plane over which interpolation is to be done
    lineDef={'start':[1.4,2.3],    #coordinates of the line's starting point in the q1-q2 plane
             'end':[-0.7,-0.2],    #coordinates of the line's end point in the q1-q2 plane
             'noPtsLine':100
            }
    #-----------------------------------------------------------------------------
    # (1) Create nodal sets over the parameter space (each node=one joint sample)    
    # Generate Gauss-Legendre points over qBounds[0] and qBounds[1]
    qNodes=[]
    for i in range(2):
        xi,wXI=gpce.GaussLeg_ptswts(nNodes[i])
        qNodes_=gpce.mapFromUnit(xi,qBound[i]) 
        qNodes.append(qNodes_)
    # Response at the GL samples
    fNodes=analyticTestFuncs.fEx2D(qNodes[0],qNodes[1],'type1','tensorProd')

    #(2) Interpolate from the nodal set to the points over the line defined above
    qLine,fLine=lagrangeInterpol_Quads2Line(fNodes,qNodes,lineDef)

    #(3) Plot results
    plt.figure(figsize=(8,5))
    plt.plot(qLine[:,0],fLine,'-ob',label='Lagrange Interpolation')
    # exact response
    fLine_ex=analyticTestFuncs.fEx2D(qLine[:,0],qLine[:,1],'type1','pair')
    plt.plot(qLine[:,0],fLine_ex,'-xr',label='Exact')
    plt.title('%d x%d interpolating nodes in Q1xQ2 space.' %(nNodes[0],nNodes[1]))
    plt.xlabel('q1');
    plt.ylabel('Response')
    plt.legend()
    plt.grid()
    plt.show()

