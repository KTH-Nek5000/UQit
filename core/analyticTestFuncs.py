######################################################################
# Test Functions for gPCE (different paramter space dimensions)
######################################################################
#
import numpy as np
import math as mt
#
#
#///////////////////
def fEx1D(z,typ):  
    """ 
       Analytical test function for 1D parameter space
    """
    z = np.array(z, copy=False, ndmin=1)
    if typ=='type1':
       val=(10.0+.7*np.sin(5.0*z)+3.*np.cos(z))
       val=np.array(val)
    elif typ=='type2':
       a=[1.,-0.2,3.0]
       val=a[0]+a[1]*z+a[2]*z**2.
    return val  

#/////////////////////////
def fEx1D_moments(Q,distType):
    """ Mean and Variance of fEx1D(z) when 
        1. z~U[qBound], U=uniform distribution, Q=qBound ('type1' in fEx1D)
        2. z~N(m,v^2), N=Normal distribution, Q=[m,v] ('type2' in fEx1D)
    """
    def mean1_(q_):
        return (10.*q_-0.7*mt.cos(5.*q_)/5.0+3.*mt.sin(q_))
    def var1_(q_):
        tmp=100*q_+0.245*(q_-0.1*mt.sin(10*q_))+4.5*(q_+0.5*mt.sin(2*q_))
        return (tmp-2.8*mt.cos(5*q_)+60*mt.sin(q_)-2.1*(mt.cos(6*q_)/6.+0.25*mt.cos(4*q_)))
    def mean2_(q_,a,b,c):
        m=q_[0]
        v=q_[1]
        return (a+b*m+c*m**2.+0.5*c*v**2.)
    def var2_(q_,a,b,c):
        m=q_[0]
        v=q_[1]
        s1=a**2.+2*a*b*m+m**2.*(b**2.+2*a*c)+2*b*c*(m**3.+2*m**2.*v)+c**2*m**4.
        s2=b**2.+2*a*c+2*b*c*m*v**2+6*m**2*v**2
        s3=c**2*v**4.
        return(s1+0.5*s2+1.5*s3)
    if distType=='Unif':
       fMean=(mean1_(Q[1])-mean1_(Q[0]))/(Q[1]-Q[0])                 
       fVar =(var1_(Q[1])-var1_(Q[0]))/(Q[1]-Q[0])-fMean**2.
    elif distType=='Norm':
       a=[1.,-0.2,3.]
       fMean=mean2_(Q,a[0],a[1],a[2])      
       fVar=var2_(Q,a[0],a[1],a[2])      
    return fMean,fVar
#
#////////////////////
def fEx2D(z1,z2,typ,method):
    """ 
       Analytical test function for 2D parameter space
       z1: n1
       z2: n2
       if method==tensorProd:
          f: n1*n2
       elif method== pair:
          (z1,z2) contains pairs of samples
          n1=n2
          f: n1 
    """
    def funVal(z1_,z2_,typ):
        if typ=='type1': # from: https://se.mathworks.com/help/symbolic/graphics.html?s_tid=CRUX_lftnav
           tmp1=3.*mt.exp(-(z2_+2)**2.-z1_**2) * (z1_-1.0)**2.
           tmp2=-(mt.exp(-(z1_+2)**2.-z1_**2))/3.
           tmp3=mt.exp(-(z1_**2.+z2_**2.))*(10.*z1_**3.-2.*z1_+10.*z2_**5.)
           tmp=tmp1+tmp2+tmp3
        elif typ=='type2':
           tmp1=3.*mt.exp(-(z2_+1)**2.-z1_**2) * (z1_-1.0)**2.
           tmp2=-1.*mt.exp(-(z2_-1.)**2.-z1_**2) * (z1_-1.)**2.
           tmp=tmp1+tmp2+0.001
        elif typ=='type3':   #simple enough for analytical derivation of Sobol indices
           tmp=z1_**2.+z1_*z2_
        elif typ=='Rosenbrock':
           tmp=100*(z2_-z1_**2.)**2.+(1-z1_)**2.
        return tmp

    z1 = np.array(z1, copy=False, ndmin=1)
    z2 = np.array(z2, copy=False, ndmin=1)
    n1=z1.shape[0]
    n2=z2.shape[0]
    f=[]
    if (method=='tensorProd'):
       for i2 in range(n2):
           z2_=z2[i2]
           for i1 in range(n1):
               z1_=z1[i1]
               k=i2*n1+i1
               tmp=funVal(z1_,z2_,typ)
               f.append(tmp)
    elif (method=='pair'):
       if (n1!=n2):
          print('ERROR in fEx2D: pairs of a paramater sample vector should have the same size')
       for i in range(n1):
           z1_=z1[i]
           z2_=z2[i]
           tmp=funVal(z1_,z2_,typ)
           f.append(tmp)
    else:
       print('ERROR in fEx2D: invalid method.')
    f=np.asarray(f)
    return f


#////////////////////
def fEx3D(z1,z2,z3,typ,method,opts):
    """ 
       Analytical test function for 3D parameter space
       z1: n1
       z2: n2
       z3: n3
       type='Ishigami'
       opts={a:aVal,b:bVal} in case of Ishigami
       if method==tensorProd:
          f: n1*n2*n3
       elif method== pair:
          (z1,z2,z3) contains tuples of samples
          n1=n2=n3
          f: n1 
    """
    def funVal(z1_,z2_,z3_,typ):
        if typ=='Ishigami': # Ishigami Function
           a=opts['a']
           b=opts['b']
           tmp=mt.sin(z1_)+a*(mt.sin(z2_))**2.+b*z3_**4.*mt.sin(z1_)
        else:
           print('ERROR in fEX3D: invalid type.')
           #new functions to be added here
        return tmp

    z1 = np.array(z1, copy=False, ndmin=1)
    z2 = np.array(z2, copy=False, ndmin=1)
    z3 = np.array(z3, copy=False, ndmin=1)
    n1=z1.shape[0]
    n2=z2.shape[0]
    n3=z3.shape[0]
    f=[]
    if (method=='tensorProd'):
       for i3 in range(n3):
           z3_=z3[i3]
           for i2 in range(n2):
               z2_=z2[i2]
               for i1 in range(n1):
                   z1_=z1[i1]
                   k=(i3*n2*n1)*(i2*n1)+i1
                   tmp=funVal(z1_,z2_,z3_,typ)
                   f.append(tmp)
    elif (method=='pair'):
       if (n1!=n2 or n1!=n3 or n2!=n3):
          print('ERROR in fEx3D: tuples of a paramater sample vector should have the same size')
       for i in range(n1):
           z1_=z1[i]
           z2_=z2[i]
           z3_=z3[i]
           tmp=funVal(z1_,z2_,z3_,typ)
           f.append(tmp)
    else:
       print('ERROR in fEx3D: invalid method.')
    f=np.asarray(f)
    return f

#///////////////////////////////////
def ishigami_exactMoments(a1,a2,a3,opts):
    """ 
       Analytical values of mean and variance of Ishigami function f(q1,q2,q3)
       Assume q1~U[a11,a12], q3~U[q21,q22], q3~U[a31,a32], 
       Assume q1,q2,q3 are mutually independent
       ai: [ai1,ai2] list specifying range of each of the 3 parameters
       opts={a:aVal,b:bVal}: parameters of Ishigami function
    """
    #evaluate the integral for the exact mean
    a1_=a1[1]-a1[0]
    a2_=a2[1]-a2[0]
    a3_=a3[1]-a3[0]
    m=-a2_*a3_*(mt.cos(a1[1])-mt.cos(a1[0]))+0.5*opts['a']*a1_*a3_*(a2_-0.5*(mt.sin(2.*a2[1])-mt.sin(2*a2[0])))-0.2*opts['b']*a2_*(a3[1]**5.-a3[0]**5.)*(mt.cos(a1[1])-mt.cos(a1[0]))
    m=(m/(a1_*a2_*a3_))  #analytical mean
    #evaluate the integral for the exact variance
    SIN2_1=(mt.sin(2*a1[1])-mt.sin(2*a1[0]))
    SIN2_2=(mt.sin(2*a2[1])-mt.sin(2*a2[0]))
    SIN4_2=(mt.sin(4*a2[1])-mt.sin(4*a2[0]))
    v1=0.5*(a1_-0.5*SIN2_1)*(a3_+opts['b']**2.*(a3[1]**9.-a3[0]**9.)/9.0 + 0.4*opts['b']*(a3[1]**5.-a3[0]**5.))*a2_
    v2=-opts['a']*(mt.cos(a1[1])-mt.cos(a1[0]))*(a2_-0.5*SIN2_2)*(a3_+0.2*opts['b']*(a3[1]**5.-a3[0]**5.))
    v3=0.25*opts['a']**2.*a1_*a3_*(1.5*a2_-SIN2_2+0.125*SIN4_2)
    v=v1+v2+v3
    v=v/(a1_*a2_*a3_) - m**2.  #analytical variance 
    return m,v
