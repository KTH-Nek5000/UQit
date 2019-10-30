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
def lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,method,GType):
    """
       Given fVal1 at nSampMod1 arbitrary samples over space1, construct a Lagrange interpolation from these and predict the values at nSampMod2 Gauss quadrature points over space2. Note that space2<space1. Also the destination Gauss quadratures can be the abscissa of different types of polynomials such as Lagrange, ..., as eventually meant to be specifid by GType. 
       - This function is useful when we want to construct a gPCE over space2, while the samples q1 are not .e.g. Gauss-Legendre points over space1. 
       - NOTE: Dimension of admissible spaces space1 and space2 should be the same =p 
           ndim: dimension of parameter spaces space1 and space2 (=p)
           qM1: List of GL samples in p-dimensions, model1: qM1=[qM1_1|qM1_2|...|qM1_p] where qM1_i are 1D numpy arrays of length nM1_i, where i=1,2,...,p. 
           fVal1: numpy pD array of size (nM1_1,nM1_2,...,nM1_p)
           spaceM1: Admissible space of qM1 = List of p 1D lists: [spaceM1_1|spaceM1_2|...|spaceM1_p] where spaceM1_i are lists [.,.]. 
           nM2=List of number of Gauss samples qM2 i npD parameter space: nM2=[nM2_1,nM2_2,...,nM2_p]
           spaceM2: Admissible space of qM2 = List of p 1D lists: [spaceM2_1|spaceM2_2|...|spaceM2_p] where spaceM1_i are lists [.,.]. 
           method: how to handle the multi-dimsnsionality of the parameters, default is 'tensorProd' 
           GType: type of Gauss points: 'GL': Gauss Legendre
            
    """
    #(1) Check if the inputs have sent in correctly
    #  (1a) Check both space1 and space2 have the same dimension
    ndim=len(spaceM1)   #dimension of parameter spaces
    if (ndim!=len(spaceM2) or ndim!=len(qM1)):
       print('ERROR in lagIntAtGaussPts: parameter spaces and samples should have the same dimensions.')
    #  (1b) Check space2<space1 in each parameter direction
    for idim in range(ndim):
       d1=spaceM1[idim][1]-spaceM1[idim][0]
       d2=spaceM2[idim][1]-spaceM2[idim][0]
       if (d2>d1):
          print('ERROR in lagIntAtGaussPts: spaceM2 should be smaller than spaceM1. Issue in parmeter %d' %(idim+1))
    #(2) Construct the Gauss-Legendre stochastic samples for model2
    qM2=[]
    for i in range(ndim):
        [xi,w]=gpce.GaussLeg_ptswts(nM2[i])   #for i-th param
        qM2_=gpce.mapFromUnit(xi,spaceM2[i])
        qM2.append(qM2_)
    if (ndim==1): 
       qM2=qM2[0]
    elif (ndim==2):
       if (method=='tensorProd'):
          qM2=reshaper.vecs2grid(qM2[0],qM2[1]) #Make a grid out of two 1D vectors
       else:
          print('ERROR in lagIntAtGaussPts: currently only tensor-product is available')
    elif(ndim==3):
       if (method=='tensorProd'):
          qM2=reshaper.vecs2grid(qM2[0],qM2[1],qM2[2]) #Make a grid out of three 1D vectors
       else:
          print('ERROR in lagIntAtGaussPts: currently only tensor-product is available')
    else:
       print('ERROR in lagIntAtGaussPts: currently up to 3D parameter space can be handled.')

    #(3) Use lagrange interpolation to find values at q2, given fVal1 at q1
    if ndim==1:
       fVal2Interp=lagrangeInterpol.lagrangeInterpol_singleVar(fValM1,qM1,qM2)
    elif (ndim==2 or ndim==3):
       fVal2Interp=lagrangeInterpol.lagrangeInterpol_multiVar(fValM1,qM1,qM2,method)
    else:
       print('ERROR in lagIntAtGaussPts: currently up to 3D parameter space can be handled.')
    return qM2,fVal2Interp

    

#/////////////////////////////////////////////////////////
def pce2pce_GaussLeg_1d(fValM1,qM1,spaceM1,nM2,spaceM2,GType):
    """
       Given fValM1 at nM1 Gauss-Legndre samples over spaceM1, construct a new stochastic collocation surrogate over 1D spaceM2 with nM2 Gauss-Legendre points. Note that spaceM2<spaceM1.
       (A Lagrange interpolation based on fValM1 at nodal set qM1 is constructed in spaceM1. Then we use it to interpolate at qM2 GL samples over spaceM2. Using GL points 2, a new PCE is constructed over space2)
       NOTE: qM1, qM2 are over their admissible spaces (not on [-1,+1])
       fValM1: numpy 1D array of length n1
       qM1: numpy 1D array of length n1
       nM2: integer
       spaceM1, spaceM2: list of two values specifying bounds [.,.]
       GType: type of Gauss points: 'GL': Gauss Legendre
    """
    #(1) Use lagrange interpolation to find values at q2 (Gauss points over space2), given fVal1 at q1
    qM2,fVal2Interp=lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,'',GType)
    #(2) Construct PCE2 over space2
    fCoef2,fMean2,fVar2=gpce.pce_LegUnif_1d_cnstrct(fVal2Interp)  
    return fCoef2,fMean2,fVar2,qM2,fVal2Interp

#////////////////////////////////////////////////////////
def pce2pce_GaussLeg_2d(fValM1,qM1,spaceM1,nM2,spaceM2,method,GType):
    """
       Given fValM1 at Gauss-Legndre samples over 2D spaceM1, construct a new stochastic collocation surrogate over the 2D spaceM2 with nM2_1xnM2_2 Gauss-Legendre points. Note that space2<space1.
       (A Lagrange interpolation based on fValM1 at nodal set qM1 is constructed in spaceM1. Then we use it to interpolate at qM2 GL samples over spaceM2. Based on GL points of M2 a new PCE is constructed over spaceM2).
       NOTE: Multi-dimensionality is handled by method that is by default 'tensorProd'

          qM1: List of GL samples in 2D, model1: qM1=[qM1_1|qM1_2] where qM1_1 and qM1_2 are 1d numpy array of lengths nM1_1 and nM1_2, respectively 
          fVal1: numpy 2D array of length (nM1_1,nM1_2)
          spaceM1: Admissible space of qM1, List of 2 one-D lists: [spaceM1_1|spaceM1_2] where spaceM1_i=[.,.]
          nM2=List of number of GL samples in the two dimensions of qM2: [nM2_1,nM2_2]
          spaceM2: Admissible space of qM2, List of 2 one-D lists: [spaceM2_1|spaceM2_2] where spaceM2_i=[.,.]
       GType: type of Gauss points: 'GL': Gauss Legendre
    """
    #(1) Use lagrange interpolation to find values at q2 (Gauss points over space2), given fVal1 at q1
    qM2,fVal2Interp=lagIntAtGaussPts(fValM1,qM1,spaceM1,nM2,spaceM2,'tensorProd',GType)
    #(2) Construct PCE2 over space2
    fCoef2,fMean2,fVar2=gpce.pce_LegUnif_2d_cnstrct(fVal2Interp,nM2[0],nM2[1])
    return fCoef2,fMean2,fVar2,qM2,fVal2Interp



############################
# External Funcs: Tests
############################
#//////////////////////////////
def pce2pce_GaussLeg_1d_test():
    """
       Test pce2pce_GaussLeg_1d(...)      
    """
    #------ SETTINGS --------------------
    nSampMod1=[7]        #number of samples in PCE1
    space1=[[-0.5,2.]]   #admissible space of param in PCE1
    nSampMod2=[5]        #number of samples in PCE2
    space2=[[0.0,1.5]]   #admissible space of param in PCE2
    nTest=100   #number of test samples
    #------------------------------------
    #(1) Construct PCE1
    [xi1,w1]=gpce.GaussLeg_ptswts(nSampMod1[0])   #Gauss sample pts in [-1,1]
    q1=gpce.mapFromUnit(xi1,space1[0])    #map Gauss points to param space
    fVal1=analyticTestFuncs.fEx1D(q1)  #function value at the parameter samples (Gauss quads)
    fCoef1,fMean1,fVar1=gpce.pce_LegUnif_1d_cnstrct(fVal1)  #find PCE coefficients

    #(2) Construct PCE2 given values predicted by PCE1 at nSampMod2 GL samples over space2
    fCoef2,fMean2,fVar2,q2,fVal2=pce2pce_GaussLeg_1d(fVal1,q1,space1,nSampMod2,space2,'GL')

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
    plt.plot(q1,fVal1,'ob',markersize=8,label='GL Samples1')
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
       Test pce2pce_GaussLeg_2d(...)      
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
    nTest=[100,100]   #number of test samples of parameter 1,2
    #---------------------------------------------------------------------
    #(1) Construct PCE1
    #GL points for param1,2
    qM1=[];
    for i in range(2):
       [xi,wXI]=gpce.GaussLeg_ptswts(nSampM1[i])   #Gauss sample pts in [-1,1]
       qM1.append(gpce.mapFromUnit(xi,spaceM1[i]))    #map Gauss points to param space
    #Response values at the GL points
    fValM1=analyticTestFuncs.fEx2D(qM1[0],qM1[1],'type1','tensorProd') 
    #Construct the PCE
    fCoefM1,fMeanM1,fVarM1=gpce.pce_LegUnif_2d_cnstrct(fValM1,nSampM1[0],nSampM1[1])  

    #(2) Construct PCE2 given values predicted by PCE1 at GL samples over spaceM2
    fCoefM2,fMeanM2,fVarM2,qM2,fVal2Interp=pce2pce_GaussLeg_2d(fValM1,qM1,spaceM1,nSampM2,spaceM2,'tensorProd','GL')

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
    fPCETestM1=gpce.pce_LegUnif_2d_eval(fCoefM1,nSampM1[0],nSampM1[1],xiTestM1[0],xiTestM1[1])

    #Predictions by PCE2 in spaceM2
    qTestM2=[]
    xiTestM2=[]
    for i in range(2):
        qTestM2_=np.linspace(spaceM2[i][0],spaceM2[i][1],nTest[i])  #test points in param1
        qTestM2.append(qTestM2_)
        xiTestM2_=gpce.mapToUnit(qTestM2[i],spaceM2[i])
        xiTestM2.append(xiTestM2_)
    fTestM2=analyticTestFuncs.fEx2D(qTestM2[0],qTestM2[1],'type1','tensorProd')   #exact response at test points of model2
    fPCETestM2=gpce.pce_LegUnif_2d_eval(fCoefM2,nSampM2[0],nSampM2[0],xiTestM2[0],xiTestM2[1])

    #(4) 2d contour plots
    plt.figure(figsize=(20,8))
    plt.subplot(1,3,1)
    ax=plt.gca()
    q1Test_1_Grid,q2Test_2_Grid,fTest_Grid=plot2d.plot2D_gridVals(qTestM1[0],qTestM1[1],fTest)  #reformat for 2D contour plot
    CS1 = plt.contour(q1Test_1_Grid,q2Test_2_Grid,fTest_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Exact Response Surface')

    plt.subplot(1,3,2)
    ax=plt.gca()
    q1TestM1_1_Grid,q2TestM1_2_Grid,fPCETestM1_Grid=plot2d.plot2D_gridVals(qTestM1[0],qTestM1[1],fPCETestM1)  #reformat for 2D contour plot
    CS2 = plt.contour(q1TestM1_1_Grid,q2TestM1_2_Grid,fPCETestM1_Grid,35)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    qM1Grid=reshaper.vecs2grid(qM1[0],qM1[1])
    plt.plot(qM1Grid[:,0],qM1Grid[:,1],'o',color='r',markersize=6)
    plt.plot(qM2[:,0],qM2[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE1')

    plt.subplot(1,3,3)
    ax=plt.gca()
    q1TestM2_1_Grid,q2TestM2_2_Grid,fPCETestM2_Grid=plot2d.plot2D_gridVals(qTestM2[0],qTestM2[1],fPCETestM2)  #reformat for 2D contour plot
    CS3 = plt.contour(q1TestM2_1_Grid,q2TestM2_2_Grid,fPCETestM2_Grid,20)#,cmap=plt.get_cmap('viridis'))
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qM2[:,0],qM2[:,1],'s',color='b',markersize=6)
    plt.xlabel('q1');plt.ylabel('q2');
    plt.title('Response Surface by PCE2 \n (Response values at GL2 samples are \n Lagrange-interpolated from PCE1)')
    plt.xlim(spaceM1[0][:])
    plt.ylim(spaceM1[1][:])
    plt.show()

