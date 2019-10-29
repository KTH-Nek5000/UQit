######################################################################
# Test Functions for gPCE (different paramter space dimensions)
######################################################################

import numpy as np
import math as mt

#///////////////////
def fEx1D(z):  
    """ 
       Analytical test function for 1D parameter space
    """
    z = np.array(z, copy=False, ndmin=1)
    val=(10.0+.7*np.sin(5.0*z)+3.*np.cos(1.*z))
    val=np.array(val)
    return val  

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
           tmp1=3.*mt.exp(-(z2_+2)**2.-z1_**2) * (z1_-1.0)**2.
           tmp2=3.*mt.exp(-(z2_-1.5)**2.-z1_**2) * (z1_-1.)**2.
           tmp=tmp1+tmp2
        elif typ=='type3':   #simple enough for analytical derivation of Sobol indices
           tmp=z1_**2.+z1_*z2_
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

