==================
UQ Forward Problem
==================

In UQ forward (uncertainty propagation) problems, the aim is to estimate the propagation of uncertainties in the model response or QoIs, from known uncertain parameters/inputs. 
In many situations, we are mostly interested in approximately estimating the statistical moments of the model outputs and QoIs. 
In particular, among the moments, our main focus is on the expected value :math:`\mathbb{E}_\mathbf{q}[r]` and variance :math:`\mathbb{V}_\mathbf{q}[r]` of a QoI :math:`r`, with :math:`\mathbf{q}` denoting the uncertain parameters.
Different approaches can be used for this purpose, see [Smith13]_, [Ghanem17]_.
The Monte Carlo approaches rely on taking independent samples from the parameters and running the model simulator as a blackbox. 
However, the convergence rate of these methods can be low, see e.g. [Ghanem17]_.
As a more efficient technique, the spectral-based method non-intrusive generalized polynomial chaos expansion (gPCE) [Xiu02]_, [Xiu05]_, [Xiu07]_ is employed in :code:`UQit` for UQ forward problems. 
For an overview of the technique with the same notations as in this document, refer to [Rezaeiravesh20]_.

.. _gPCE-sect:

Standard Polynomial Chaos Expansion
-----------------------------------

Theory
~~~~~~
The generalized polynomial chaos expansion for :math:`\tilde{f}(\chi,\mathbf{q})` is written as,

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=0}^K \hat{f}_k(\chi) \Psi_{k}(\xi) \,,

where the basis function for the multi-variate parameter :math:`\xi\in\Gamma` is defined as :math:`\Psi_{k}(\xi)=\prod_{i=1}^p \psi_{k_i}(\xi_i)`.
In the framework of generalized PCE, see [Xiu02]_, given the distribution of the single-variate parameter :math:`\xi_i\in \Gamma_i`, a set of orthogonal basis functions are chosen for :math:`\psi_{k_i}(\xi_i)`, for :math:`i=1,2,\ldots,p`. 
Note that, there is a one-to-one correspondence between any sample of :math:`\mathbf{q}\in \mathbb{Q}` and :math:`\xi\in\Gamma`, where :math:`\mathbb{Q}=\bigotimes_{i=1}^p \mathbb{Q}_i` and :math:`\Gamma=\bigotimes_{i=1}^p \Gamma_i`.
The mapped space :math:`\Gamma_i` is known based on the gPCE rule, see [Xiu02]_, [Eldred09]_.


Given a set of training data :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}_{i=1}^n`, there are two main steps to construct the above expansion.
First, a truncation scheme is needed to handle :math:`p`-dimensional parameter space and determine :math:`K`.
Currently, tensor-product and total-order schemes are available in :code:`UQit`. 
Second, the coefficients :math:`\{\hat{f}_k(\chi)\}_{k=0}^K` have to be determined. 
In :code:`UQit`, two different approaches can be used for this purpose: projection and regression method, see [Rezaeiravesh20]_ and the references therein. 
In case the number of training data is less than :math:`K`, compressed sensing method can be adopted which is implemented in :code:`UQit` through the external Python library :code:`cvxpy` [Diamond16]_.

Once the coefficients :math:`\{\hat{f}_k(\chi)\}_{k=0}^K` are obtained, the PCE can be used as a surrogate for the actual unobserved :math:`f(\chi,\mathbf{q})`.
A main advantage of the PCE method is that the approximate estimation of the statistical moments of the :math:`f(\chi,\mathbf{q})` or response :math:`r` is a natural outcome of the surrogate construction. 
Using gPCE, the mean and variance of the simulator are estimated by,

.. math::
   \mathbb{E}_{\mathbf{q}}[f(\chi,\mathbf{q})] = \hat{f}_0(\chi),

.. math::
   \mathbb{V}_{\mathbf{q}}[f(\chi,\mathbf{q})] = \sum_{k=1}^K \hat{f}^2_k(\chi) \gamma_k, 

where :math:`\gamma_k` is the inner-product of the polynomial basis.


Example
~~~~~~~
In :code:`UQit`, to construct and estimate expected value and variance of :math:`f(\mathbf{q})` for :math:`\mathbf{q}\in\mathbb{Q}\subset \mathbb{R}^p`, we have:

.. code-block:: python

   pce_=pce(fVal=fVal,xi=xiGrid,pceDict=pceDict,nQList=nQ)
   fMean=pce_.fMean       
   fVar=pce_.fVar         
   pceCoefs=pce_.coefs     
   kSet=pce_.kSet

To evaluate a constructed PCE at a set of test parameter samples taken from :math:`\Gamma`, we write:

.. code-block:: python

   pcePred_=pce.pceEval(coefs=pceCoefs,xi=xiTest,distType=distType,kSet=kSet)
   fPCE=pcePred_.pceVal

As for instance described in [Rezaeiravesh20]_, as an a-posteriori measure of the convergence of the PCE terms, we can evaluate the following indicator,

.. math::
   \vartheta_\mathbf{k} = |\hat{f}_\mathbf{k}| \, \|\Psi_{\mathbf{k}}(\mathbf{\xi})\|_2/|\hat{f}_0|
   
at different multi-indices :math:`\mathbf{k}`. 
In :code:`UQit` this is done through running,

.. code-block:: python

   pce.convPlot(coefs=pceCoefs,distType=distType)



Implementation
~~~~~~~~~~~~~~
In :code:`UQit`, the methods required for standard PCE are implemented in :code:`pce.py`. 

.. automodule:: pce
   :members:

Notebook
~~~~~~~~
Try the `PCE notebook <../examples/pce.ipynb>`_ to see how to use :code:`UQit` to perform standard polynomial chaos expansion (PCE). The provided examples can also be seen as a way to validate the implementation of the methods in :code:`UQit`.

Probabilistic Polynomial Chaos Expansion
----------------------------------------

Theory
~~~~~~
The standard PCE (polynomial chaos expansion) and GPR (Gaussian process regression) are two powerful approaches for surrogate construction and metamodeling in UQ. 
Combining these two approaches, probabilistic PCE is derived. 
There are at least two different views to this derivation which can be found in Schobi et al. [Schobi15]_ and Owen [Owen17]_. 
In :code:`UQit`, a generalization of the latter is implemented which is detailed in [Rezaeiravesh20]_. 


Example
~~~~~~~
Given training parameter samples :code:`qTrain` and associated responses :code:`yTrain` with observation noise :code:`noiseSdev`, the :code:`ppce` is constructed as follows. 

.. code-block:: python

   ppce_=ppce.ppce(qTrain,yTrain,noiseSdev,ppceDict)
   optOut=ppce_.optOut
   fMean_samples=ppce_.fMean_samps
   fVar_samples=ppce_.fVar_samps


Implementation
~~~~~~~~~~~~~~
.. automodule:: ppce
   :members:


Notebook
~~~~~~~~
Try this `PPCE notebook <../examples/ppce.ipynb>`_ to see how to use :code:`UQit` to perform probabilistic polynomial chaos expansion (PPCE). 


