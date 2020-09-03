==================
UQ Forward Problem
==================

In UQ forward problems, the aim is to estimate the distribution of the model response or QoIs, given known distributions for the uncertain parameters. 
Hence, these problems are also called uncertain propagation problems. 
In many situations, we are mostly intersted in estimating the statsitical moments of the model outputs and QoIs rather than constructing their distributions. 
In particular among the moments, our main focus is on estimating the expected value :math:`\mathbb{E}_\mathbf{q}[r]` and varaince :math:`\mathbb{V}_\mathbf{q}[r]` of a QoI :math:`r` for uncertain parameters :math:`\mathbf{q}`.

There are different appraoches for UQ forward problem, see [Smith,uqHandbook]. 
The Monte Carlo approaches relying on taking independent parameter samples and blackbox treatment of the model response play a pivotal role. 
However, the convergence rate of these methods can be low, see e.g. [Smith].
As a substitute in :code:`UQit`, we use spectral methods which represent the stochastic nature of a UQ problem.
The particular focus is kept on non-intrusive polynomial chaos expansion [???] with two standard and probabilistic types, as explained in detail in the following parts. 
For more theoretical background and relevant literature, refer to [Rezaeiravesh et al.].


Standard Polynomial Chaos Expansion
-----------------------------------

Theory
~~~~~~

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=0}^K \hat{f}_k(\chi) \Psi_{k}(\mathbf{q})


The basis function for the multi-variate parameter :math:`\mathbf{q}` is defined as :math:`\Psi_{k}(\mathbf{q})=\prod_{i=1}^p \psi_{k_i}(q_i)`.
In the framework of generlized PCE, see [????], given the distribution of a single-variate parameter :math:`q_i`, a set of orthogonal basis function is chosen as the first step to construct a PCE. 
In the non-intrusive PCE, where there are a limited number of training data :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}`, there are two more main steps to construct the above expansion.
First a tructation scheme is needed to handle :math:`p`-dimensional parameter space. 


Finally, the coefficients :math:`\{\hat{f}_k(\chi)\}_{k=0}^K` have to be determined. 
In :code:`UQit`, two different approaches can be used for this purpose: projection and regression method, see e.g. [Smith,Rezairavesh:20,uqHandbook]. 
In the projection method, the coefficients in the PCE are computed through a numerical integration by Gauss quarature rule. 
This demands, the :math:`n_i` parameter samples in the :math:`i`-th dimension to be the zeros of the polynomial basis :math:`\psi_{k_i}(q_i)`. 
Given the samples for each of the :math:`p` parameters, a :math:`p`-dimensional grid is constructed using tensor product.
As a result of using the tensor-product truncation scheme, :math:`K+1=\prod_{i=1}^p n_i`.

In contrast, in regression method there is no obligation on how the parameter samples are taken from the space of the paramters.
However, for sake of increasing the accuracy of the expansion, a space-filling sampling method is required.
To handle multi-dimensionality, both tensor-product or total-order truncation schemes can be employed. 
The latter leads to :math:`K+1=(L+p)!/L!p!` where :math:`L` is the maximum polynomial order in each direction. 
In the regression method, an algebraic linear system is numerically solved to estimate coeffcients 
:math:`\{\hat{f}_k(\chi)\}_{k=0}^K`.
If numbe of training samples :math:`n` is not less than :math:`K+1`, the system is determined or over-determined and is iteratively solved. 
But, in :code:`UQit`, it is allowed that :math:`n<(K+1)` which results in an under-determined linear system. 
To find a unique solution to such problem, compressed sensing method [???] is adopted which is implemented through using the external Python library :code:`cvxpy`, see `here <https://www.cvxpy.org/>`_.

Once the coeffcients :math:`\{\hat{f}_k(\chi)\}_{k=0}^K` are obtained, the PCE can be used as a surrogate for actual unobserved `f(\chi,\mathbf{q})`.
An advantage of the PCE surrogate is that the approximate estimation for the statsitical moments of the `f(\chi,\mathbf{q})` is the natural outcome of the surrogate construction. 
The mean and varaince of the simulator are estimated by,

.. math::
   \mathbb{E}_{\mathbf{q}}[f(\chi,\mathbf{q})] = \hat{f}_0(\chi),

.. math::
   \mathbb{V}_{\mathbf{q}}[f(\chi,\mathbf{q})] = \sum_{k=1}^K \hat{f}^2_k(\chi) \gamma_k, 

where :math:`\gamma_k` is the inner-product of the polynomail basis.



   


Implementation
~~~~~~~~~~~~~~
In :code:`UQit`, the methods required for standard PCE are implemented in :code:`pce.py`. 


.. automodule:: pce
   :members:



Example
~~~~~~~
To construct and estimate expected value and variance of :math:`f(q)` for :math:`q\in\mathbb{Q}\subset \mathbb{R}`:

.. code-block:: python

   pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
   fMean=pce_.fMean       #E[f(q)]
   fVar=pce_.fVar         #V[f(q)]
   pceCoefs=pce_.coefs    
   kSet=pce_.kSet

Notebook
~~~~~~~~
Try this `PCE notebook <../examples/pce.ipynb>`_ to see how to use :code:`UQit` to perform standard polynomial chaos expansion (PCE). The provided examples can also be seen as a way to validate the implementation of the methods in :code:`UQit`.




Probabilistic Polynomial Chaos Expansion
----------------------------------------

Theory
~~~~~~

Implementation
~~~~~~~~~~~~~~
.. automodule:: ppce
   :members:

Example
~~~~~~~

Notebook
~~~~~~~~
