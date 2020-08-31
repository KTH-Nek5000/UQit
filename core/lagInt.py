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
#
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
#
class lagInt:
   """
   Lagrange interpolation over a p-D parameter space.     
   """
   def __init__(self,qNodes,fNodes,qTest,liDict=[]):
       self.qNodes=qNodes
       self.fNodes=fNodes
       self.qTest=qTest
       self.liDict=liDict
       self._info()
       self.interp()
    
   def _info(self):
       """
       Checks the consistency in the inputs
       """
       self.p=len(self.qNodes)
       if self.p>1:
          if 'testRule' not in self.liDict.keys():
             raise KeyError("For p>1, 'testRule' ('tensorProd' or 'set') should be set in 'liDict.") 
          else:
             testRule=self.liDict['testRule']
             if testRule!='tensorProd' and testRule!='set':
                raise ValueError("Invalid value for 'tensorProd'") 

   def interp(self):
       """
       Lagrange interpolation, for p=1,2, ...
       """
       if self.p==1:
          self.interp_1d()              
       elif self.p>1:
          self.interp_pd() 
          
   def basis1d(self,qNodes_,k,Q_):
      """
      Constructs single-variate Lagrange bases :math:`L_k(q)` using `n` nodes `qNodes_`. 
      Note that `k=0,1,...,n-1`. The bases are evaluated at `m` test points `Q_`, 
      for `p=1` (single-variate parameter). 

      Args:
        `qNodes_`: 1D numpy array of size n
           A single-variate training nodal set
        `Q_`: 1D numpy array of size m
           Test samples 
        `k`: int, 1D numpy array
           Order of the polynomial bases

      Returns:
         `prod`: n-by-m numpy array
            Values of :math:`L_k(Q)` for `k=0,1,...,n-1` evaluated at `Q_`
      """
      n=qNodes_.size
      m=Q_.shape[-1]
      prod=np.zeros((k.shape[-1],m))
      for k_ in k:
          prod_=1.0
          for j in range(n):
              if j!=k_:
                 prod_*=(Q_-qNodes_[j])/(qNodes_[k_]-qNodes_[j])
          prod[k_,:]=prod_       
      return prod

   def interp_1d(self):
      R"""
      Lagrange interpolation of order (n-1) constructed over a 1D parameter space.
      Bases are constructed from n nodes `qNodes` and are evaluated at test points `Q` in `[min(qNodes_),max(qNodes_)]`. 

      .. math::
         F(Q)=\sum_{k=0}^{n-1} fNodes_{k} L_k(Q)

      where, Lagrange Bases L_k(q) are constructed from the nodal set.

      Args:
        `qNodes`: List of length 1
           `qNodes=[qNodes_1]` where `qNodes_1` is a 1D numpy array of size n containing 
           the parameter's training nodal set.
        `qTest`: List of length 1 of a 1D numpy array of size m
           Test samples 
           NOTE: make sure `qTest` falls in `[min(qNodes),max(qNodes)]`
        `fNodes`: 1D numpy array of size n  
           Simulator's response value at the training nodes

      Returns:
        `val`: 1D numpy array of size m
           Value of f(qTest)
      """  
      qNodes=self.qNodes[0]
      fNodes=self.fNodes
      Q=self.qTest[0]
      if (np.all(Q>max(qNodes))):
         raise ValueError('qTest cannot be larger than max(qNodes).')
      if (np.all(Q<min(qNodes))):
         raise ValueError('qTest cannot be smaller than min(qNodes).')
      if (fNodes.size!=qNodes.size):
         raise ValueError('qNodes and fNodes should have the same size.') 
      n=fNodes.size
      k=np.arange(n)
      Lk=self.basis1d(qNodes,k,Q)
      fInterp=np.matmul(fNodes[None,:],Lk).T
      self.val=fInterp[:,0]

   def interp_pd(self):
      R"""
      Lagrange interpolation of order :math:`(n_1-1)*(n_2-1)*...*(n_p-1)` constructed over a 
      p-D parameter space. Here, :math:`n_k, k=1,2,...,p` refers to the number of training nodes in the i-th dimension of the parameter space.

      .. math::
          F(\mathbf{Q})=\sum_{k_1=1}^{n_1} ... \sum_{k_p=1}^{n_p} [fNodes(k_1,k_2,...,k_p) L_{k_1}(Q_1) L_{k_2}(Q_2) ... L_{k_=}(Q_p)]
         
      where, :math:`L_{k_i}(Q_p)` is the single-variate Lagrange basis in the i-th dimension. 

      Args:      
        `qNodes`: A list of length p
           `qNodes=[qNodes_1|qNodes_2|...|qNodes_p]`, where `qNodes_k` is a 1D numpy array 
            of `n_k` nodes in the k-th parameter dimension. 
        `fNodes`: a p-D numpy array of shape `(n_1,n_2,...,n_p)` 
           (or a 1D array of length `n=n_1*n_2*...*n_p`)
           Response values at the training nodes.
        `qTest`: A list of length p containing test samples from the p-D parameter space.      
           `qTest=[Q_0,Q_1,...,Q_{p-1}]` where `Q_i` is a 1D numpy array of size `m_i, i=0,1,...,p-1`
           Note that `Q_i` must be in `[min(qNodes_i),max(qNodes_i)]` for `i=1,2,...,p`.
        `liDict`: A dict to set different options for Lagrange interpolation over the p-D space, with the following key:
           * 'testRule': The rule for treating the multi-dimensional test points with the values:
             - 'tensorProd': 
               a tensor-product grid is constructed from `Q_i` in list `qTest`.
               Hence, total number of sample points `m=m0*m1*m_{p-1}`
             - 'set': 
               All `m_i` for `i=0,1,...p-1` should be equal. A
               set of `m=m_i` test points is considered.
      Returns: 
        `val`: f(qTest), Interpolated values of f(q) at `qTest`
           * if 'testRule'=='tensorProd' => `val`: pD numpy array of shape `(m1,m2,...,mp)`
           * if 'testRule'=='set' => 'val': 1D numpy array of size `m1*m2*...*mp`
      """
      Q=np.asarray(self.qTest)
      qNodes=self.qNodes
      fNodes=self.fNodes
      p=self.p
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
      testRule=self.liDict['testRule']
      if (p!=len(Q)):
          raise ValueError('qNodes and qTest should be of the same dimension.')
      for i in range(p):
         if (np.all(Q[i]>np.amax(qNodes[i],axis=0))):
            raise ValueError('qTest cannot be larger than max(qNodes) in %d-th dim.'%i)
         if (np.all(Q[i]<np.amin(qNodes[i],axis=0))):
            raise ValueError('qTest cannot be smaller than min(qNodes) in %d-th dim.'%i)
         if (fNodes.shape[i]!=nList[i]):
            raise ValueError('qNodes and fNodes should be of the same size.')
      #Construct and evaluate Lagrange interpolation
      idxTest=[]    #List of indices counting the test points
      Lk=[]         #List of Lagrange bases
      for i in range(p):
          idxTest.append(np.arange(mList[i]))
          k=np.arange(nList[i])
          Lk_=self.basis1d(qNodes[i],k,Q[i]) #Basis at the i-th dim
          Lk.append(Lk_)

      if testRule=='tensorProd':
         idxTestGrid=reshaper.vecs2grid(idxTest)    
      elif testRule=='set':
         idxTestGrid=idxTest[0]   #same for all dimensions

      mTot=idxTestGrid.shape[0]   #total number of test points
      fInterp=np.zeros(mTot)
      if p>2:
         mulInd=[[i for i in range(p-1,-1,-1)]]*(p-1)   #list of indices for tensordot
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
      self.val=fInterp
#
def lagInt_Quads2Line(fNodes,qNodes,lineDef):
    """ 
    First constructing a Lagrange interpolation from tensor-product nodal set in a 2D parameter space and then evaluating it at the test points located on a straight line lying in the same parameter plane.

    Args: 
      `qNodes`: a list of length 2
         `qNodes=[Q1|Q2]` list of training nodes `Qi` is a 1D numpy array of length ni for i=1,2.       
      `fNodes`: 1D numpy array of length `n1*n2`
         Values of the response at `qNodes`
      `lineDef`: a dictionary defining the line over which test samples are taken.
         ={`'lineStart':[q1Start,q2Start]`,    line's starting point
           `'ineEnd':[q1End,q2End]`,           line's end point 
           `'noPtsLine':<int>`}                 number of test points on the line          
    Returns:       
      `val`: f(qTest), Interpolated values of f(q) at `qTest`
         1D numpy array of size `m1*m2*...*mp`
    """
    p=len(qNodes)
    nQ=[qNodes[0].shape[0],qNodes[1].shape[0]]
    if (fNodes.ndim==1):
       fNodes=np.reshape(fNodes,nQ,'F')  
    #limits of the parameters space
    qBound=[[min(qNodes[0]),max(qNodes[0])],
            [min(qNodes[1]),max(qNodes[1])]]
    #(1) Check if the line is relying on the parameters plane
    lineStart=lineDef['start'] 
    lineEnd  =lineDef['end'] 
    for i in range(p):
        if (lineStart[i]<qBound[i][0] or lineStart[i]>qBound[i][1]):
           print(i,lineStart[i],qBound[i][0],lineStart[i],qBound[i][1]) 
           raise ValueError('Test line cannot be outside of the training plane. Check lineStart in dim %d' %i)
        if (lineEnd[i]<qBound[i][0] or lineEnd[i]>qBound[i][1]):
           raise ValueError('Test line cannot be outside of the training plane. Check lineEnd in dim %d' %i)
          
    #(2) Generate interpolation points over the line
    noPtsLine=lineDef['noPtsLine']
    q1=np.linspace(lineStart[0],lineEnd[0],noPtsLine)
    slope=(lineEnd[1]-lineStart[1])/(lineEnd[0]-lineStart[0])
    q2=slope*(q1-lineEnd[0])+lineEnd[1]

    #(3) Lagerange interpolation from quadratures to the points on the line
    qLine=[q1,q2]
    fLine=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qLine,liDict={'testRule':'set'}).val
    return qLine,fLine
#
#
# Tests
#
def lagInt_test_1d():
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
def lagInt_test_2d():
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
def lagInt_test_3d():
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
