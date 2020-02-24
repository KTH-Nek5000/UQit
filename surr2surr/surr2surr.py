###############################################################
#     Interpolate from a Surrogate To Another Surrogate
###############################################################
# We have a surrogate that is over Q1 admissibile space. We want
#    to use it to construct a new surrogate over Q2\subsetQ1. 
#
#--------------------------------------------------------------
# Saleh Rezaeiravesh, salehr@kth.se
#--------------------------------------------------------------
import sys
import numpy as np	
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../gPCE/')
sys.path.append('../analyticFuncs/')
sys.path.append('../lagrangeInterpol/')
sys.path.append('../general/')
sys.path.append('../plot/')
import gpce
import analyticTestFuncs
import lagrangeInterpol
import reshaper
import plot2d


#////////////////////////////////////////////////////////////////////
def lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,GType):
    """
       Given fValM1 at nM1 arbitrary samples over spaceM1, construct a Lagrange interpolation from these and predict the values at nM2 Gauss quadrature points over spaceM2. Note that ||spaceM2||<||spaceM1|| at each dimension. Also the destination Gauss quadratures can be the abscissa of different types of polynomials such as Lagrange, ..., as eventually meant to be specifid by GType. 
       - This function is useful when we want to construct a gPCE over spaceM2, while the samples qM1 are not e.g. Gauss-Legendre points over spaceM1. 
       - This function is currently working for 1d, 2d, 3d parameter spaces.
       - NOTE: Dimension of admissible spaces spaceM1 and spaceM2 should be the same =p 
           ndim: dimension of parameter spaces spaceM1 and spaceM2 (=p)
           qM1: List of samples (non-uniform) in p-dimensions for model1: qM1=[qM1_1|qM1_2|...|qM1_p] where qM1_i are 1D numpy arrays of length nM1_i, where i=1,2,...,p. 
           fValM1: numpy pD array of size (nM1_1,nM1_2,...,nM1_p)
           spaceM1: Admissible space of qM1 = List of p 1D lists: [spaceM1_1|spaceM1_2|...|spaceM1_p] where spaceM1_i is a list of two elements [.,.]. 
           nM2=List of the number of Gauss quadratures qM2 in pD parameter space: nM2=[nM2_1,nM2_2,...,nM2_p]
           spaceM2: Admissible space of qM2 = List of p 1D lists: [spaceM2_1|spaceM2_2|...|spaceM2_p] where spaceM1_i is a list of two elements [.,.]. 
           pDmethod: how to handle the multi-dimsnsionality of the parameters, default is 'tensorProd' 
           GType: type of Gauss quadratures: 'GL': Gauss Legendre
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
    #(2) Construct the Gauss-Legendre stochastic samples for model2
    qM2=[]
    xiM2=[]
    for i in range(ndim):
        [xi_,w]=gpce.GaussLeg_ptswts(nM2[i])   #for i-th param
        qM2_=gpce.mapFromUnit(xi_,spaceM2[i])
        qM2.append(qM2_)
        xiM2.append(xi_)
    if (ndim==1): 
       qM2=qM2[0]
    elif (ndim==2):
       if (pDmethod=='tensorProd'):
          qM2=reshaper.vecs2grid(qM2[0],qM2[1]) #Make a grid out of two 1D vectors
          xiM2=reshaper.vecs2grid(xiM2[0],xiM2[1]) #Make a grid out of two 1D vectors
       else:
          print('ERROR in lagIntAtGaussPts: currently only tensor-product is available')
    elif(ndim==3):
       if (pDmethod=='tensorProd'):
          qM2=reshaper.vecs2grid3d(qM2[0],qM2[1],qM2[2]) #Make a grid out of three 1D vectors
          xiM2=reshaper.vecs2grid3d(xiM2[0],xiM2[1],xiM2[2]) #Make a grid out of three 1D vectors
       else:
          print('ERROR in lagIntAtGaussPts: currently only tensor-product is available')
    else:
       print('ERROR in lagIntAtGaussPts: currently up to 3D parameter space can be handled.')

    #(3) Use lagrange interpolation to find values at q2, given fVal1 at q1
    if ndim==1:
       fVal2Interp=lagrangeInterpol.lagrangeInterpol_singleVar(fValM1,qM1[0],qM2)
    elif (ndim==2 or ndim==3):
       fVal2Interp=lagrangeInterpol.lagrangeInterpol_multiVar(fValM1,qM1,qM2,pDmethod)
    else:
       print('ERROR in lagIntAtGaussPts: currently up to 3D parameter space can be handled.')
    return qM2,xiM2,fVal2Interp

#////////////////////////////////////////////////////////
def pce2pce_GaussLeg(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,GType):
    """
       Given fValM1 at Gauss-Legendre samples over pD spaceM1, construct a new PCE over the pD spaceM2 with nM2_1xnM2_2x...xnM2_p (i.e. tensor product) Gauss-Legendre points. Note that ||spaceM2||<||space1|| at each of the p dimensions.
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
           GType: type of Gauss points: 'GL': Gauss Legendre
    """
    #(1) Check if the inputs have sent in correctly
    #  (1a) Check both space1 and space2 have the same dimension
    ndim=len(spaceM1)   #dimension of parameter spaces
    if (ndim!=len(spaceM2) or ndim!=len(qM1)):
       print('ERROR in pce2pce_GaussLeg: parameter spaces and samples should have the same dimensions.')
    #  (1b) Check ||spaceM2<||spaceM1|| in each parameter direction
    for idim in range(ndim):
       d1=spaceM1[idim][1]-spaceM1[idim][0]
       d2=spaceM2[idim][1]-spaceM2[idim][0]
       if (d2>d1):
          print('ERROR in pce2pce_GaussLeg: ||spaceM2|| should be smaller than ||spaceM1||. Issue in parmeter %d' %(idim+1))
    #(2) Use lagrange interpolation to find values at qM2 (Gauss points over spaceM2), given fValM1 at Gauss-Legendre samples qM1
    qM2,xiGridM2,fVal2Interp=lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,pDmethod,GType)
    #(3) Construct PCE2 over spaceM2
    if ndim==1:
       fCoef2,fMean2,fVar2=gpce.pce_LegUnif_1d_cnstrct(fVal2Interp)  
    else: #multi-dimensional param space
       pceDict={'sampleType':'GQ','pceSolveMethod':'Projection','truncMethod':'TO'}
       pceDict=gpce.pceDict_corrector(pceDict)
       if ndim==2:
          fCoef2,kSet2,fMean2,fVar2=gpce.pce_LegUnif_2d_cnstrct(fVal2Interp,[nM2[0],nM2[1]],xiGridM2,pceDict)
       elif ndim==3:  
          fCoef2,kSet2,fMean2,fVar2=gpce.pce_LegUnif_3d_cnstrct(fVal2Interp,[nM2[0],nM2[1],nM2[2]],xiGridM2,pceDict)
       else:
          print('ERROR in pce2pce_GaussLeg: currently up to 3D parameter space can be handled')
    return fCoef2,kSet2,fMean2,fVar2,qM2,fVal2Interp


############################
# External Funcs: Tests
############################
#//////////////////////////////
def pce2pce_GaussLeg_1d_test():
    """
       Test pce2pce_GaussLeg(...) for 1 uncertain parameter     
    """
    #------ SETTINGS --------------------
    nSampMod1=[7]        #number of samples in PCE1
    space1=[[-0.5,2.]]   #admissible space of param in PCE1
    nSampMod2=[5]        #number of samples in PCE2
    space2=[[0.0,1.5]]   #admissible space of param in PCE2
    nTest=100   #number of test samples
    #------------------------------------
    #(1) Construct PCE1
    q1=[]
    [xi1,w1]=gpce.GaussLeg_ptswts(nSampMod1[0])   #Gauss sample pts in [-1,1]
    q1_=gpce.mapFromUnit(xi1,space1[0])    #map Gauss points to param space
    q1.append(q1_)
    fVal1=analyticTestFuncs.fEx1D(q1[0])  #function value at the parameter samples (Gauss quads)
    fCoef1,fMean1,fVar1=gpce.pce_LegUnif_1d_cnstrct(fVal1)  #find PCE coefficients

    #(2) Construct PCE2 given values predicted by PCE1 at nSampMod2 GL samples over space2
    fCoef2,kSet2,fMean2,fVar2,q2,fVal2=pce2pce_GaussLeg(fVal1,q1,space1,nSampMod2,space2,'','GL')

    #(3) Make predictions by PCE1 and PCE2 over their admissible spaces
    qTest1=np.linspace(space1[0][0],space1[0][1],nTest)  #test points in param space
    fTest1=analyticTestFuncs.fEx1D(qTest1)   #exact response at test points
    xiTest1=gpce.mapToUnit(qTest1,space1[0])
    fPCETest1=gpce.pce_LegUnif_1d_eval(fCoef1,xiTest1)    

    qTest2=np.linspace(space2[0][0],space2[0][1],nTest)  #test points in param space
    fTest2=analyticTestFuncs.fEx1D(qTest2)   #exact response at test points
    xiTest2=gpce.mapToUnit(qTest2,space2[0])
    fPCETest2=gpce.pce_LegUnif_1d_eval(fCoef2,xiTest2)    

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

#//////////////////////////////
def pce2pce_GaussLeg_2d_test():
    """
       Test pce2pce_GaussLeg(...) for 2D uncertain parameter space
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
    #(1) Construct PCE1
    #GL points for param1,2
    qM1=[];
    xiM1=[]
    for i in range(2):
       [xi_,wXI]=gpce.GaussLeg_ptswts(nSampM1[i])   #Gauss sample pts in [-1,1]
       qM1.append(gpce.mapFromUnit(xi_,spaceM1[i]))    #map Gauss points to param space
       xiM1.append(xi_)
  
    #Response values at the GL points
    fValM1=analyticTestFuncs.fEx2D(qM1[0],qM1[1],'type1','tensorProd') 
    #Construct the PCE
    pceDict={'sampleType':'GQ','pceSolveMethod':'Regression','truncMethod':'TO','LMax':10}
    pceDict=gpce.pceDict_corrector(pceDict)
    xiGridM1=reshaper.vecs2grid(xiM1[0],xiM1[1])
    fCoefM1,kSetM1,fMeanM1,fVarM1=gpce.pce_LegUnif_2d_cnstrct(fValM1,[nSampM1[0],nSampM1[1]],xiGridM1,pceDict)  

    #(2) Construct PCE2 given values predicted by PCE1 at GL samples over spaceM2
    fCoefM2,kSetM2,fMeanM2,fVarM2,qM2,fVal2Interp=pce2pce_GaussLeg(fValM1,qM1,spaceM1,nSampM2,spaceM2,'tensorProd','GL')

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
        xiTestM1_=gpce.mapToUnit(qTestM1[i],spaceM1[i])
        xiTestM1.append(xiTestM1_)
    fPCETestM1=gpce.pce_LegUnif_2d_eval(fCoefM1,kSetM1,xiTestM1[0],xiTestM1[1])

    #Predictions by PCE2 in spaceM2
    qTestM2=[]
    xiTestM2=[]
    for i in range(2):
        qTestM2_=np.linspace(spaceM2[i][0],spaceM2[i][1],nTest[i])  #test points in param1
        qTestM2.append(qTestM2_)
        xiTestM2_=gpce.mapToUnit(qTestM2[i],spaceM2[i])
        xiTestM2.append(xiTestM2_)
    fTestM2=analyticTestFuncs.fEx2D(qTestM2[0],qTestM2[1],'type1','tensorProd')   #exact response at test points of model2
    fPCETestM2=gpce.pce_LegUnif_2d_eval(fCoefM2,kSetM2,xiTestM2[0],xiTestM2[1])

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
    qM1Grid=reshaper.vecs2grid(qM1[0],qM1[1])
    plt.plot(qM1Grid[:,0],qM1Grid[:,1],'o',color='r',markersize=6)
    plt.plot(qM2[:,0],qM2[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE1')

    plt.subplot(1,3,3)
    ax=plt.gca()
    fPCETestM2_Grid=fPCETestM2.reshape((qTestM2[0].shape[0],qTestM2[1].shape[0]),order='F').T
    CS3 = plt.contour(qTestM2[0],qTestM2[1],fPCETestM2_Grid,20)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qM2[:,0],qM2[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE2 \n (Response values at GL2 samples are \n Lagrange-interpolated from PCE1)')
    plt.xlim(spaceM1[0][:])
    plt.ylim(spaceM1[1][:])
    plt.show()

