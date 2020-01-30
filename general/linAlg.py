##########################################################
#   Linear Algebra tools
##########################################################
#  Saleh Rezaeiravesh, salehr@kth.se
#---------------------------------------------------------
import numpy as np
import cvxpy as cvx

#/////////////////////////////
def myLinearRegress(A,R,L_=1):
    """
       Solve a linear system of equations.
       This solver works for uniquely-, over-, and under-determined linear system. 
       If system is under-determined we use compressed sensing with L1 or L2 regularization. Library cvxpy is used for this purpose.  https://www.cvxpy.org
          Input:
             Af=R => At*A f = At*R (normal system)
             A:nxK (n:no of data, K: no of PCE terms), f: unknown coefs:Kx1, R: Responses nx1
             L_: =1 or 2: Regularization Order
          Output:
             set of coeffcients f: 1d array of length K
    """
    def linearSysSolver(A,R,L_=1):
        """
           Linear Regression
        """
        n=A.shape[0]   #number of data
        K=A.shape[1]   #number of unknowns
        #make the system normal
        M=np.dot(A.T,A)   
        R=np.dot(A.T,R)
#        if (K>n): #only under-determined system => use compressed sensing
        if (0==0): #always use compressed sensing
           f = cvx.Variable(K)
           objective = cvx.Minimize(cvx.norm(f, L_))   #L1/L2-regularization
           constraints = [M*f == R]
           prob = cvx.Problem(objective, constraints)
           object_value = prob.solve(verbose=True)
           print('...... Compressed sensing (regularization) is done to compute PCE coeffcients, fHat.')
           print('       Min objective value=||fHat||= %g in L%d-sense.'%(object_value,L_))
           fHat=f.value
        else:
           fHat=np.linalg.solve(M,R) 
        return fHat
    f=linearSysSolver(A,R,1)
    return f

