#######################################################################
# Lagrange Interpolation is constructed based on tensor-product rule 
#  using the response values at a given nodal set and then it is used to 
#  interpolate at some test points within the parameter space.
#######################################################################
# Saleh Rezaeiravesh, salehr@kth.se
#-----------------------------------------------------------------------
#
"""
Note: 
  * To avoid Runge's phenomenon, the training nodal set should be non-uniformly distributed in each dimension of the parameter space. 

"""
import os
import sys
import numpy as np
import UQit.reshaper as reshaper
#
#
class lagInt:
   """
   Lagrange interpolation over a p-D parameter space, where p=1,2,...
   The interpolation order is :math:`(n_1-1)*(n_2-1)*...*(n_p-1)`, where, 
   :math:`n_k, k=1,2,...,p` refers to the number of training nodes in the i-th dimension of the parameter space.

   .. math::
      F(\mathbf{Q})=\sum_{k_1=1}^{n_1} ... \sum_{k_p=1}^{n_p} [fNodes(k_1,k_2,...,k_p) L_{k_1}(Q_1) L_{k_2}(Q_2) ... L_{k_p}(Q_p)]
         
   where, :math:`L_{k_i}(Q_i)` is the single-variate Lagrange basis in the i-th dimension. 

   Args:      
      `qNodes`: A list of length p
         `qNodes=[qNodes_1,qNodes_2,...,qNodes_p]`, 
         where `qNodes_k` is a 1D numpy array of `n_k` nodes in the k-th parameter dimension. 
      `fNodes`: A p-D numpy array of shape `(n_1,n_2,...,n_p)` (or a 1D array of size `n=n_1*n_2*...*n_p` if p>1)
          Response values at the training nodes.
      `qTest`: A list of length p 
          Containing test samples from the p-D parameter space, i.e. `qTest=[Q_0,Q_1,...,Q_{p-1}]` 
          where `Q_i` is a 1D numpy array of size `m_i, i=0,1,...,p-1`.
          Note that `Q_i` must be in `[min(qNodes_i),max(qNodes_i)]` for `i=1,2,...,p`.
      `liDict`: dict (optional)
          To set different options for Lagrange interpolation over the p-D space when p>1, with the following key:
            * 'testRule': The rule for treating the multi-dimensional test points with the values:
               - 'tensorProd': A tensor-product grid is constructed from `Q_i`s in list `qTest`.
                 Hence, the total number of sample points is `m=m0*m1*m_{p-1}`.
               - 'set': All `m_i` for `i=0,1,...p-1` should be equal. As a result, 
                 a set of `m=m_i` test points is considered.
   Returns: 
      `val`: f(qTest), Interpolated values of f(q) at `qTest`
         * If p==1: 1D numpy array of size m1
         * If p>1 and 'testRule'=='tensorProd' => `val`: pD numpy array of shape `(m1,m2,...,mp)`
         * If p>1 and 'testRule'=='set' => 'val': 1D numpy array of size `m1*m2*...*mp`
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
      Constructs single-variate Lagrange bases :math:`L_k(q)` using `n` nodes `qNodes_` for 
      `k=0,1,...,n-1`. The bases are evaluated at `m` samples `Q_` of a single-variate parameter 

      Args:
        `qNodes_`: 1D numpy array of size n
           The single-variate training nodal set
        `Q_`: 1D numpy array of size m
           Test samples 
        `k`: 1D numpy array of int values
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
          F(\mathbf{Q})=\sum_{k_1=1}^{n_1} ... \sum_{k_p=1}^{n_p} [fNodes(k_1,k_2,...,k_p) L_{k_1}(Q_1) L_{k_2}(Q_2) ... L_{k_p}(Q_p)]
         
      where, :math:`L_{k_i}(Q_p)` is the single-variate Lagrange basis in the i-th dimension. 
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
    Constructing a Lagrange interpolation from tensor-product nodal set in a 2D parameter space and then evaluating it at the test points located on a straight line lying in the same parameter plane.

    Args: 
      `qNodes`: A list of length 2
         `qNodes=[Q1,Q2]` list of training nodes `Qi` is a 1D numpy array of length ni for i=1,2.       
      `fNodes`: 1D numpy array of length `n1*n2`
         Values of the response at `qNodes`
      `lineDef`: dict
         Defines the line over which the test samples are to be taken. The keys are:
           * `'lineStart':[q1Start,q2Start]`; line's starting point
           * `'lineEnd':[q1End,q2End]`; line's end point 
           * `'noPtsLine'`: int; number of test points on the line, `m1`
    Returns:       
      `val`: f(qTest), Interpolated values of f(q) at `qTest`
         1D numpy array of size `m1`
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

