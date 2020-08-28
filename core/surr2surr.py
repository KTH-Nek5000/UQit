###############################################################
#     Interpolate from a Surrogate To Another Surrogate
###############################################################
# We have a surrogate that is over Q1 admissibile space. We want
#    to use it to construct a new surrogate over Q2\subsetQ1. 
#
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
#TO do: go through it
#--------------------------------------------------------------
#
import os
import sys
import numpy as np	
import matplotlib
import matplotlib.pyplot as plt
UQit=os.getenv("UQit")
sys.path.append(UQit)
from pce import pce, pceEval
import analyticTestFuncs
from lagInt import lagInt
import reshaper
import plot2d
#
#
def lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,distType):
    """
       Given fValM1 at nM1 arbitrary samples over spaceM1, construct a Lagrange interpolation from these and predict the values at nM2 Gauss quadrature points over spaceM2. Note that ||spaceM2||<||spaceM1|| at each dimension. Also the destination Gauss quadratures can be the abscissa of different types of polynomials such as Legendre, ..., as eventually meant to be specifid by distType.
       - This function is useful when we want to construct a gPCE over spaceM2, while the samples qM1 are not e.g. Gauss-quadrature points over spaceM1. 
       - NOTE: Dimension of admissible spaces spaceM1 and spaceM2 should be the same =p 
           ndim: dimension of parameter spaces spaceM1 and spaceM2 (=p)
           qM1: List of samples (non-uniform) in p-dimensions for model1: qM1=[qM1_1|qM1_2|...|qM1_p] where qM1_i are 1D numpy arrays of length nM1_i, where i=1,2,...,p. 
           fValM1: numpy pD array of size (nM1_1,nM1_2,...,nM1_p)
           spaceM1: Admissible space of qM1 = List of p 1D lists: [spaceM1_1|spaceM1_2|...|spaceM1_p] where spaceM1_i is a list of two elements [.,.]. 
           nM2=List of the number of Gauss quadratures qM2 in pD parameter space: nM2=[nM2_1,nM2_2,...,nM2_p]
           spaceM2: Admissible space of qM2 = List of p 1D lists: [spaceM2_1|spaceM2_2|...|spaceM2_p] where spaceM1_i is a list of two elements [.,.]. 
           pDmethod: how to handle the multi-dimsnsionality of the parameters, default is 'tensorProd' 
           distType: distribution type of the parameters acc. gPCE rule
    """
    #(1) Check if the inputs have sent in correctly
    #  (1a) Check both space1 and space2 have the same dimension
    ndim=len(spaceM1)   #dimension of parameter spaces
    if (ndim!=len(spaceM2) or ndim!=len(qM1)):
       print('ERROR in lagIntAtGaussPts: parameter spaces and samples should have the same dimensions.')
    #  (1b) Check ||spaceM2<||spaceM1|| in each parameter direction
    for idim in range(ndim):
       d1=spaceM1[idim][1]-spaceM1[idim][0]
       d2=spaceM2[idim][1]-spaceM2[idim][0]
       if (d2>d1):
          print('ERROR in lagIntAtGaussPts: ||spaceM2|| should be smaller than ||spaceM1||. Issue in parmeter %d' %(idim+1))
    #(2) Construct the Gauss-quadratures stochastic samples for model2
    qM2=[]
    xiM2=[]
    for i in range(ndim):
        xi_,w=pce.gqPtsWts(nM2[i],'Unif')   #for i-th param
        qM2_=pce.mapFromUnit(xi_,spaceM2[i])
        qM2.append(qM2_)
        xiM2.append(xi_)
    if (ndim==1): 
       qM2=qM2[0]
    elif (ndim>1):
       if (pDmethod=='tensorProd'):
          #qM2=reshaper.vecs2grid(qM2) #Make a grid out of two 1D vectors
          xiM2=reshaper.vecs2grid(xiM2) #Make a grid out of two 1D vectors
       else:
          print('ERROR in lagIntAtGaussPts: currently only tensor-product is available')
    #(3) Use lagrange interpolation to find values at q2, given fVal1 at q1
    if ndim==1:
       fVal2Interp=lagInt(fNodes=fValM1,qNodes=[qM1[0]],qTest=[qM2]).val
    elif (ndim>1):
       fVal2Interp_=lagInt(fNodes=fValM1,qNodes=qM1,qTest=qM2,liDict={'testRule':pDmethod}).val
       nM2_=fVal2Interp_.size
       fVal2Interp=fVal2Interp_.reshape(nM2_,order='F')
    return qM2,xiM2,fVal2Interp
#
def pce2pce_GQ(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,distType):
    """
       Given fValM1 at training samples over pD spaceM1, construct a new PCE over the pD spaceM2 with nM2_1xnM2_2x...xnM2_p (i.e. tensor product) points. Note that ||spaceM2||<||space1|| at each of the p dimensions.
       This is how it is done: A Lagrange interpolation based on fValM1 at nodal set qM1 is constructed in spaceM1. Then the Lagrange interpolation is exploited to predict values at qM2 GL samples over spaceM2. Finally, based on the qM2 GL points a new PCE model M2 is constructed over spaceM2).
       - NOTE: Multi-dimensionality is handled by pDmethod that is by default 'tensorProd'. Other methods can be implemented in future. 
       - NOTE: This function is currently working for 1d, 2d, 3d parameter spaces.
       - NOTE: Dimension of admissible spaces spaceM1 and spaceM2 should be the same =p 
           ndim: dimension of parameter spaces spaceM1 and spaceM2 (=p)
           qM1: List of GL samples in p-dimensions for model1: qM1=[qM1_1|qM1_2|...|qM1_p] where qM1_i are 1D numpy arrays of length nM1_i, where i=1,2,...,p. 
           fValM1: numpy pD array of size (nM1_1,nM1_2,...,nM1_p)
           spaceM1: Admissible space of qM1 = List of p 1D lists: [spaceM1_1|spaceM1_2|...|spaceM1_p] where spaceM1_i is a list of two elements [.,.]. 
           nM2=List of the number of Gauss samples qM2 in pD parameter space: nM2=[nM2_1,nM2_2,...,nM2_p]
           spaceM2: Admissible space of qM2 = List of p 1D lists: [spaceM2_1|spaceM2_2|...|spaceM2_p] where spaceM1_i is a list of two elements [.,.]. 
           pDmethod: how to handle the multi-dimsnsionality of the parameters, default is 'tensorProd' 
           distType: distribution type of the parameters acc. gPCE rule
    """
    #(1) Check if the inputs have sent in correctly
    #  (1a) Check both space1 and space2 have the same dimension
    ndim=len(spaceM1)   #dimension of parameter spaces
    if (ndim!=len(spaceM2) or ndim!=len(qM1)):
       print('ERROR in pce2pce_GQ: parameter spaces and samples should have the same dimensions.')
    #  (1b) Check ||spaceM2<||spaceM1|| in each parameter direction
    for idim in range(ndim):
       d1=spaceM1[idim][1]-spaceM1[idim][0]
       d2=spaceM2[idim][1]-spaceM2[idim][0]
       if (d2>d1):
          print('ERROR in pce2pce_GQ: ||spaceM2|| should be smaller than ||spaceM1||. Issue in parmeter %d' %(idim+1))
    #(2) Use lagrange interpolation to find values at qM2 (Gauss points over spaceM2), given fValM1 at Gauss-Legendre samples qM1
    qM2,xiGridM2,fVal2Interp=lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,distType)
    #(3) Construct PCE2 over spaceM2
    if ndim==1:
        pceDict={'p':ndim,'sampleType':'GQ','pceSolveMethod':'Projection','distType':distType} 
        pce_=pce(fVal=fVal2Interp,xi=[],pceDict=pceDict)    
        kSet2=[]
    elif (ndim>1): #multi-dimensional param space
        pceDict={'p':ndim,'sampleType':'GQ','pceSolveMethod':'Projection','truncMethod':'TP',
                 'distType':distType}
        pce_=pce(fVal=fVal2Interp,xi=xiGridM2,pceDict=pceDict,nQList=nM2)
        kSet2=pce_.kSet        
    fMean2=pce_.fMean
    fVar2=pce_.fVar
    fCoef2=pce_.coefs
    return fCoef2,kSet2,fMean2,fVar2,qM2,fVal2Interp
#
#
# Tests
#
def pce2pce_GQ_1d_test():
    """
       Test pce2pce_GQ(...) for 1 uncertain parameter     
    """
    #------ SETTINGS --------------------
    nSampMod1=[7]        #number of samples in PCE1
    space1=[[-0.5,2.]]   #admissible space of param in PCE1
    nSampMod2=[5]        #number of samples in PCE2
    space2=[[0.0,1.5]]   #admissible space of param in PCE2
    nTest=100   #number of test samples
    #------------------------------------
    distType='Unif'      #distribution type of the RV
    #(1) Construct PCE1
    q1=[]
    xi1,w1=pce.gqPtsWts(nSampMod1[0],distType)   #Gauss sample pts in [-1,1]
    q1_=pce.mapFromUnit(xi1,space1[0])    #map Gauss points to param space
    q1.append(q1_)
    fVal1=analyticTestFuncs.fEx1D(q1[0],'type1')  #function value at the parameter samples (Gauss quads)
    pceDict={'p':1,'sampleType':'GQ','pceSolveMethod':'Projection','distType':distType} 
    pce_=pce(fVal=fVal1,xi=[],pceDict=pceDict)    
    fMean1=pce_.fMean  
    fVar1=pce_.fVar
    fCoef1=pce_.coefs

    #(2) Construct PCE2 given values predicted by PCE1 at nSampMod2 GL samples over space2
    fCoef2,kSet2,fMean2,fVar2,q2,fVal2=pce2pce_GQ(fVal1,q1,space1,nSampMod2,space2,'',distType)

    #(3) Make predictions by PCE1 and PCE2 over their admissible spaces
    qTest1=np.linspace(space1[0][0],space1[0][1],nTest)  #test points in param space
    fTest1=analyticTestFuncs.fEx1D(qTest1,'type1')   #exact response at test points
    xiTest1=pce.mapToUnit(qTest1,space1[0])
    pcePred_=pceEval(coefs=fCoef1,xi=xiTest1,distType=distType)
    fPCETest1=pcePred_.pceVal

    qTest2=np.linspace(space2[0][0],space2[0][1],nTest)  #test points in param space
    fTest2=analyticTestFuncs.fEx1D(qTest2,'type1')   #exact response at test points
    xiTest2=pce.mapToUnit(qTest2,space2[0])
    pcePred_=pceEval(coefs=fCoef2,xi=xiTest2,distType=distType)
    fPCETest2=pcePred_.pceVal

    #(4) Plot
    plt.figure(figsize=(15,8))
    plt.plot(qTest1,fTest1,'--k',lw=2,label=r'Exact $f(q)$')
    plt.plot(qTest1,fPCETest1,'-b',lw=2,label='PCE1')
    plt.plot(qTest2,fPCETest2,'-r',lw=2,label='PCE2')
    plt.plot(q1[0],fVal1,'ob',markersize=8,label='GL Samples1')
    plt.plot(q2,fVal2,'sc',markersize=8,label='GL Samples2')
    plt.xlabel(r'$q$',fontsize=26)
    plt.ylabel(r'$f(q)$',fontsize=26)
    plt.title('PCE1 with GL1-pts is given. Lagrange interpolation is constructed from GL1 points to predict response values at GL2 points. Then PCE2 is constrcuted. ')
    plt.grid()
    plt.legend()
    plt.show()
#
def pce2pce_GQ_2d_test():
    """
       Test pce2pce_GQ(...) for 2D uncertain parameter space
    """
    #------ SETTINGS ----------------------------------------------------
    #PCE Model 1
    nSampM1=[6,7]        #number of samples in PCE1, parameter 1,2
    spaceM1=[[-2,1.5],   #admissible space of PCE1 (both parameters)
             [-3,2.5]]
    #PCE Model 2
    nSampM2=[4,5]        #number of samples in PCE2, parameter 1,2
    spaceM2=[[-0.5,1],   #admissible space of PCEw (both parameters)
             [-2.,1.5]]
    #Test samples
    nTest=[100,101]   #number of test samples of parameter 1,2
    #---------------------------------------------------------------------
    p=2
    distType=['Unif','Unif']
    #(1) Construct PCE1
    #GL points for param1,2
    qM1=[];
    xiM1=[]
    for i in range(p):
       xi_,wXI=pce.gqPtsWts(nSampM1[i],distType[i])   #Gauss sample pts in [-1,1]
       qM1.append(pce.mapFromUnit(xi_,spaceM1[i]))    #map Gauss points to param space
       xiM1.append(xi_)
  
    #Response values at the GL points
    fValM1=analyticTestFuncs.fEx2D(qM1[0],qM1[1],'type1','tensorProd') 
    #Construct the PCE
    pceDict={'p':p,'sampleType':'GQ','pceSolveMethod':'Projection','truncMethod':'TP','LMax':10,
             'distType':distType}
    xiGridM1=reshaper.vecs2grid(xiM1)
    pce_=pce(fVal=fValM1,xi=xiM1,pceDict=pceDict,nQList=nSampM1)
    fMeanM1=pce_.fMean
    fVarM1=pce_.fVar
    fCoefM1=pce_.coefs
    kSetM1=pce_.kSet
    
    #(2) Construct PCE2 given values predicted by PCE1 at GL samples over spaceM2
    fCoefM2,kSetM2,fMeanM2,fVarM2,qM2,fVal2Interp=pce2pce_GQ(fValM1,qM1,spaceM1,nSampM2,spaceM2,'tensorProd',distType)

    #(3) Make predictions by PCE1 and PCE2 over their admissible spaces
    #Predictions by PCE1 over spaceM1
    qTestM1=[]
    for i in range(2):
        qTestM1_=np.linspace(spaceM1[i][0],spaceM1[i][1],nTest[i])  #test points in param_i
        qTestM1.append(qTestM1_)
    fTest =analyticTestFuncs.fEx2D(qTestM1[0],qTestM1[1],'type1','tensorProd')   #exact response at test points of model1
    #GL points
    xiTestM1=[]
    for i in range(2):
        xiTestM1_=pce.mapToUnit(qTestM1[i],spaceM1[i])
        xiTestM1.append(xiTestM1_)
    pcePred_=pceEval(coefs=fCoefM1,xi=xiTestM1,distType=distType,kSet=kSetM1)
    fPCETestM1=pcePred_.pceVal

    #Predictions by PCE2 in spaceM2
    qTestM2=[]
    xiTestM2=[]
    for i in range(2):
        qTestM2_=np.linspace(spaceM2[i][0],spaceM2[i][1],nTest[i])  #test points in param1
        qTestM2.append(qTestM2_)
        xiTestM2_=pce.mapToUnit(qTestM2[i],spaceM2[i])
        xiTestM2.append(xiTestM2_)
    fTestM2=analyticTestFuncs.fEx2D(qTestM2[0],qTestM2[1],'type1','tensorProd')   #exact response at test points of model2
    pcePred_=pceEval(coefs=fCoefM2,xi=xiTestM2,distType=distType,kSet=kSetM2)
    fPCETestM2=pcePred_.pceVal

    #(4) 2d contour plots
    plt.figure(figsize=(20,8))
    plt.subplot(1,3,1)
    ax=plt.gca()
    fTest_Grid=fTest.reshape((qTestM1[0].shape[0],qTestM1[1].shape[0]),order='F').T
    CS1 = plt.contour(qTestM1[0],qTestM1[1],fTest_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact Response Surface')

    plt.subplot(1,3,2)
    ax=plt.gca()
    fPCETestM1_Grid=fPCETestM1.reshape((qTestM1[0].shape[0],qTestM1[1].shape[0]),order='F').T
    CS2 = plt.contour(qTestM1[0],qTestM1[1],fPCETestM1_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    qM1Grid=reshaper.vecs2grid(qM1)
    plt.plot(qM1Grid[:,0],qM1Grid[:,1],'o',color='r',markersize=6)
    qM2_=reshaper.vecs2grid(qM2)
    plt.plot(qM2_[:,0],qM2_[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE1')

    plt.subplot(1,3,3)
    ax=plt.gca()
    fPCETestM2_Grid=fPCETestM2.reshape((qTestM2[0].shape[0],qTestM2[1].shape[0]),order='F').T
    CS3 = plt.contour(qTestM2[0],qTestM2[1],fPCETestM2_Grid,20)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qM2_[:,0],qM2_[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE2 \n (Response values at GL2 samples are \n Lagrange-interpolated from PCE1)')
    plt.xlim(spaceM1[0][:])
    plt.ylim(spaceM1[1][:])
    plt.show()
#    
