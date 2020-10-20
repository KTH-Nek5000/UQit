"""
Tests for pce
"""
#
import os
import sys
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.getenv("UQit"))
from pce import pce, pceEval, convPlot
import analyticTestFuncs
import writeUQ
import reshaper
import sampling
#
#
def pce_1d_test():
    """
    Test PCE for 1D uncertain parameter 
    """
    #--- settings -------------------------
    #Parameter settings
    distType='Norm'      #distribution type of the parameter
    if distType=='Unif':
       qInfo=[-2,4.0]    #parameter range only if 'Unif'
       fType='type1'     #Type of test exact model function
    elif distType=='Norm':
       qInfo=[.5,0.9]    #[m,v] for 'Norm' q~N(m,v^2)
       fType='type2'     #Type of test exact model function
    n=20                 #number of training samples
    nTest=200            #number of test sample sin the parameter space
    #PCE Options
    sampleType='GQ'     #'GQ'=Gauss Quadrature nodes
                        #''= any other sample => only 'Regression' can be selected
                        # see `trainSample` class in sampling.py
    pceSolveMethod='Projection' #'Regression': for any combination of sample points 
                                #'Projection': only for GQ
    LMax_=10            #number of terms (=K) in PCE (Only used with Regresson method)
                   #(LMax will be over written by nSamples if it is provided for 'GQ'+'Projection')
    #-------------------------------------
    #(0) Make the pceDict
    pceDict={'p':1,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,'LMax':LMax_,
             'distType':[distType]}
    #(1) Generate training data
    samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType,qInfo=qInfo,nSamp=n)
    q=samps.q
    xi=samps.xi
    qBound=samps.qBound
    fEx=analyticTestFuncs.fEx1D(q,fType,qInfo)   
    f=fEx.val
    #(2) Compute the exact moments (as the reference data)
    fEx.moments(qInfo)
    fMean_ex=fEx.mean
    fVar_ex=fEx.var
    #(3) Construct the PCE
    pce_=pce(fVal=f,xi=xi[:,None],pceDict=pceDict)
    fMean=pce_.fMean     #E[f(q)] estimated by PCE
    fVar=pce_.fVar       #V[f(q)] estimated by PCE
    pceCoefs=pce_.coefs  #Coefficients in the PCE
    #(4) Compare moments: exact vs. PCE estimations
    print(writeUQ.printRepeated('-',70))
    print('-------------- Exact -------- PCE --------- Error % ')
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_ex,fMean,(fMean-fMean_ex)/fMean_ex*100.))
    print('Var  of f(q) = %g\t%g\t%g' %(fVar_ex,fVar,(fVar-fVar_ex)/fVar_ex*100.))
    print(writeUQ.printRepeated('-',70))
    #(5) Plots
    # Plot convergence of the PCE coefficients
    convPlot(coefs=pceCoefs,distType=distType)
    #
    #(6) Evaluate the PCE at test samples
    # Test samples
    testSamps=sampling.testSample('unifSpaced',GQdistType=distType,qInfo=qInfo,qBound=qBound,nSamp=nTest)
    qTest=testSamps.q
    xiTest=testSamps.xi
    fTest=analyticTestFuncs.fEx1D(qTest,fType,qInfo).val   #exact response at test samples
    #Prediction by PCE at test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=[xiTest],distType=[distType])
    fPCE=pcePred_.pceVal    
    #Exact and PCE response surface
    plt.figure(figsize=(12,5))
    ax=plt.gca()
    plt.plot(qTest,fTest,'-k',lw=2,label=r'Exact $f(q)$')
    plt.plot(q,f,'ob',label=sampleType+' Training Samples')
    plt.plot(qTest,fPCE,'-r',lw=2,label='PCE')
    plt.plot(qTest,fMean*np.ones(len(qTest)),'-b',label=r'$\mathbb{E}[f(q)]$')
    ax.fill_between(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),color='powderblue',alpha=0.4)
    plt.plot(qTest,fMean+1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b',label=r'$\mathbb{E}[f(q)]\pm 95\%CI$')
    plt.plot(qTest,fMean-1.96*mt.sqrt(fVar)*np.ones(len(qTest)),'--b')
    plt.title('Example of 1D PCE for random variable of type %s' %distType)
    plt.xlabel(r'$q$',fontsize=19)
    plt.ylabel(r'$f(q)$',fontsize=19)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(alpha=0.3)
    plt.legend(loc='best',fontsize=17)
    plt.show()
#    
def pce_2d_test():
    """
    Test PCE for 2D uncertain parameter
    """
    #---- SETTINGS------------
    #Parameters specifications
    distType=['Unif','Norm']   #distribution type of the parameters q1, q2
    qInfo=[[-2,3],             #info on parameters
           [-2,0.4]] 
    nQ=[10,12]                   #number of training samples of parameters 
    nTest=[121,120]            #number of test points in parameter spaces to evaluate PCE
    #PCE Options
    truncMethod='TO'     #'TP'=Tensor Product
                         #'TO'=Total Order  
    sampleType=['GQ','normRand']  #'GQ'=Gauss Quadrature nodes ('Projection' or 'Regression')
                             #For other type of samples, see sampling.py, trainSample => only 'Regression' can be used
                      #'LHS': Latin Hypercube Sampling (only when all distType='Unif')
    fType='type1'#Type of the exact model response, 'type1', 'type2', 'type3', 'Rosenbrock'
    pceSolveMethod='Regression' #'Regression': for any combination of sampling and truncation methods
                                #'Projection': only for 'GQ'+'TP'
    if truncMethod=='TO':
       LMax=10   #max polynomial order in each parameter dimention
    #------------------------
    p=len(distType)
    #Assemble the pceDict       
    pceDict={'p':p,'truncMethod':truncMethod,'sampleType':sampleType,
            'pceSolveMethod':pceSolveMethod,'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax,'pceSolveMethod':'Regression'})
    #Generate the training data
    xi=[]
    q=[]
    qBound=[]
    if sampleType[0]=='LHS' and sampleType[1]=='LHS':
        if distType==['Unif']*p:
           qBound=qInfo
           xi=sampling.LHS_sampling(nQ[0]*nQ[1],[[-1,1]]*p)
           for i in range(p):
               q.append(pce.mapFromUnit(xi[:,i],qBound[i]))       
           fEx_=analyticTestFuncs.fEx2D(q[0],q[1],fType,'comp')
           xiGrid=xi
        else:  
           raise ValueError("LHS works only when all q have 'Unif' distribution.") 
    else:   
       for i in range(p):
           samps=sampling.trainSample(sampleType=sampleType[i],GQdistType=distType[i],
                   qInfo=qInfo[i],nSamp=nQ[i])
           q.append(samps.q)
           xi.append(samps.xi)
           qBound.append(samps.qBound)
       fEx_=analyticTestFuncs.fEx2D(q[0],q[1],fType,'tensorProd')
       xiGrid=reshaper.vecs2grid(xi)
    fVal=fEx_.val
    #Construct the PCE
    pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet
    #Plot the convergence indicator of the PCE
    convPlot(coefs=pceCoefs,distType=distType,kSet=kSet)
    #Generate test samples for the parameters and evaluate the exact response surface at them
    qTest=[]
    xiTest=[]
    for i in range(p):
        testSamps=sampling.testSample('unifSpaced',GQdistType=distType[i],qInfo=qInfo[i],
                qBound=qBound[i],nSamp=nTest[i])
        qTest_=testSamps.q
        xiTest_=testSamps.xi
        qTest.append(qTest_)
        xiTest.append(xiTest_)
    fTest=analyticTestFuncs.fEx2D(qTest[0],qTest[1],fType,'tensorProd').val
    #Evaluate PCE at the test samples
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
    fPCE=pcePred_.pceVal
    #Use MC method to directly estimate reference values for the mean and varaiance of f(q) 
    fEx_.moments(distType,qInfo)
    fMean_mc=fEx_.mean
    fVar_mc=fEx_.var
    
    #Compare the PCE estimates for moments of f(q) with the reference values from MC
    print(writeUQ.printRepeated('-',70))
    print('------------ MC -------- PCE --------- Error % ')
    print('Mean of f(q) = %g\t%g\t%g' %(fMean_mc,fMean,(fMean-fMean_mc)/fMean_mc*100.))
    print('Var  of f(q) = %g\t%g\t%g' %(fVar_mc,fVar,(fVar-fVar_mc)/fVar_mc*100.))
    print(writeUQ.printRepeated('-',70))
    #Plot the exact and PCE response surfaces as contours in the parameters space
    # Create 2D grid from the test samples and plot the contours of response surface over it
    fTestGrid=fTest.reshape(nTest,order='F')
    fErrorGrid=(abs(fTestGrid-fPCE))         
    # 2D grid from the sampled parameters
    if sampleType[0]=='LHS' and sampleType[1]=='LHS':
       qGrid=reshaper.vecsGlue(q[0],q[1])
    else:
       qGrid=reshaper.vecs2grid(q)
    plt.figure(figsize=(21,8));
    plt.subplot(1,3,1)
    ax=plt.gca()
    CS1 = plt.contour(qTest[0],qTest[1],fTestGrid.T,40)
    plt.clabel(CS1, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.xlabel(r'$q_1$');plt.ylabel(r'$q_2$')
    plt.title('Exact Response')
    plt.subplot(1,3,2)
    ax=plt.gca()
    CS2 = plt.contour(qTest[0],qTest[1],fPCE.T,40)
    plt.clabel(CS2, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.xlabel(r'$q_1$');plt.ylabel(r'$q_2$')
    plt.title('PCE Response')
    plt.subplot(1,3,3)
    ax=plt.gca()
    CS3 = plt.contour(qTest[0],qTest[1],fErrorGrid.T,40)
    plt.clabel(CS3, inline=True, fontsize=13,colors='k',fmt='%0.2f',rightside_up=True,manual=False)
    plt.xlabel(r'$q_1$');plt.ylabel(r'$q_2$')
    plt.plot(qGrid[:,0],qGrid[:,1],'o',color='r',markersize=7)
    plt.title('|Exact-Surrogate|')
    plt.show()
#     
def pce_3d_test():
    """
    Test PCE for 3D uncertain parameter
    """
    #----- SETTINGS------------
    distType=['Unif','Unif','Unif']   #distribution type of the parameters
    qInfo=[[-0.75,1.5],               #range of parameters
             [-0.5,2.5],   
             [ 1.0,3.0]]   
    nQ=[6,5,4]                        #number of parameter samples in the 3 dimensions
    funOpt={'a':7,'b':0.1}            #parameters in the Ishigami function
    #PCE options
    truncMethod='TO'  #'TP'=Tensor Product
                      #'TO'=Total Order  
    sampleType='GQ'   #'GQ'=Gauss Quadrature nodes
                      #other types: see trainSample in sampling.py
    pceSolveMethod='Regression' #'Regression': for any combination of sample points and truncation methods
                                #'Projection': only for 'GQ'+'TP'
    nTest=[5,4,3]   #number of test samples for the parameters                          
    if truncMethod=='TO':
       LMax=10   #max polynomial order in each parameter direction
    #--------------------
    p=len(distType)
    #Assemble the pceDict
    pceDict={'p':p,'truncMethod':truncMethod,'sampleType':sampleType,'pceSolveMethod':pceSolveMethod,
             'distType':distType}
    if truncMethod=='TO':
       pceDict.update({'LMax':LMax})
    #Generate training data
    xi=[]
    q=[]
    qBound=[]
    for i in range(p):
        samps=sampling.trainSample(sampleType=sampleType,GQdistType=distType[i],qInfo=qInfo[i],nSamp=nQ[i])
        xi.append(samps.xi)
        q.append(samps.q)
        qBound.append(samps.qBound)
    fEx=analyticTestFuncs.fEx3D(q[0],q[1],q[2],'Ishigami','tensorProd',funOpt)  
    fVal=fEx.val
    #Construct the PCE   
    xiGrid=reshaper.vecs2grid(xi)
    pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
    fMean=pce_.fMean
    fVar=pce_.fVar
    pceCoefs=pce_.coefs
    kSet=pce_.kSet
    #Convergence of the PCE terms
    convPlot(coefs=pceCoefs,distType=distType,kSet=kSet)
    #Exact moments of the Ishigami function
    fEx.moments(qInfo=qBound)
    m=fEx.mean
    v=fEx.var
    #Compare the moments estimated by PCE with the exact analytical values
    print(writeUQ.printRepeated('-',50))
    print('\t\t Exact \t\t PCE')
    print('E[f]:  ',m,fMean)
    print('V[f]:  ',v,fVar)
    print(writeUQ.printRepeated('-',50))
    #Compare the PCE predictions at test points with the exact values of the model response
    qTest=[]
    xiTest=[]
    for i in range(p):
        testSamps=sampling.testSample('unifSpaced',GQdistType=distType[i],qInfo=qInfo[i],qBound=qBound[i],nSamp=nTest[i])
        qTest.append(testSamps.q)
        xiTest.append(testSamps.xi)
    fVal_test_ex=analyticTestFuncs.fEx3D(qTest[0],qTest[1],qTest[2],'Ishigami','tensorProd',funOpt).val  
    #PCE prediction at test points
    pcePred_=pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
    fVal_test_pce=pcePred_.pceVal
    #Plot the exact and PCE response values 
    nTest_=np.prod(np.asarray(nTest))
    fVal_test_pce_=fVal_test_pce.reshape(nTest_,order='F')
    err=np.linalg.norm(fVal_test_pce_-fVal_test_ex)
    plt.figure(figsize=(10,4))
    plt.plot(fVal_test_pce_,'-ob',mfc='none',ms=5,label='Exact')
    plt.plot(fVal_test_ex,'-xr',ms=5,label='PCE')
    plt.xlabel('Index of test samples, k')
    plt.ylabel('Model response')
    plt.legend(loc='best')
    plt.grid(alpha=0.4)
    plt.show()
    print('||fEx(q)-fPCE(q)|| % = ',err*100)
#
