###########################################
#     Tools for Linear Algebra 
###########################################
#  Saleh Rezaeiravesh, salehr@kth.se
#------------------------------------------
#
import numpy as np
import cvxpy as cvx
#
def myLinearRegress(A,R,L_=1,max_iter_=100000):
    """    
    Solves the linear system of equations Af=R in its normalized form A'Af=A'R
      This solver works for uniquely-, over-, and under-determined linear systems.
      If system is under-determined, the compressed sensing method with L1 or L2 regularization is used.
      For this purpose, the library cvxpy is used (https://www.cvxpy.org).
    
    Args:
      `A`: numpy array of shape (n,K) 
         
      `R`: numpy array of size n

      `L_`: int (optional)
         Specifies the regularization order, L_=1 or 2
      `max_iter_`: int (optional)   
         Maximum number of iterations to find the optimal solution when doing compressed sensing

    Returns:     
       `f`: 1D numpy array of size K
         The solution of the linear system A.f=R 
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
#        if (K>=n): #only under-determined system => use compressed sensing
        if (K!=n): #always use compressed sensing (regularization)
           f = cvx.Variable(K)
           objective = cvx.Minimize(cvx.norm(f, L_))   #L1/L2-regularization
           constraints = [M*f == R]
           prob = cvx.Problem(objective, constraints)
           object_value = prob.solve(verbose=True,max_iter=max_iter_)
           print('...... Compressed sensing (regularization) is done.')
           print('       Min objective value=||fHat||= %g in L%d-sense.'%(object_value,L_))
           fHat=f.value
        else:
           fHat=np.linalg.solve(M,R) 
        return fHat
    f=linearSysSolver(A,R,1)
    return f

