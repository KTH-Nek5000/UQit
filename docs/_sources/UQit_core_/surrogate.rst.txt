.. _surrogates-sect:

==========
Surrogates
==========

A surrogate or metamodel is an approximation of the actual model function or simulator over the parameter space.
Running a surrogate is computationally much less expensive than the actual simulator, a characteristic that makes the use of surrogates inevitable in different UQ problems.
However, the predictions by the surrogate should be accurate enough compared to the actual predictions by the simulator. 
Using an additive error model, the following relation can be considered between the model function (simulator) and its observations:

.. math::
   r=f(\chi,\mathbf{q})+\mathbf{\varepsilon}\,,

where :math:`\mathbf{\varepsilon}` expresses the bias and random noise.
Our task is to construct a surrogate :math:`\tilde{f}(\chi,\mathbf{q})` for the actual model function (simulator).

Treating the simulator as a blackbox, a set of training data :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}_{i=1}^n` is obtained. 
There are different techniques to construct a surrogate, for instance see [Smith13]_, [Ghanem17]_, and [Gramacy20]_. 
Some of the techniques relevant to CFD applications have been implemented in :code:`UQit`.
Here, we provide a short overview to the theory behind these techniques, explain their implementation in :code:`UQit`, and provide examples to show how to use them. 


Non-intrusive Polynomial Chaos Expansion
-----------------------------------------
As a type of stochastic collocation (SC) methods, see e.g. Chapter 20 in [Ghanem17]_, non-intrusive PCE [Xiu05]_, [Xiu07]_ can be used to construct a surrogate. 

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=0}^K \hat{f}_k(\chi) \Psi_{k}(\xi) \,.

There is a one-to-one correspondence between any sample of :math:`\mathbf{q}\in \mathbb{Q}` and :math:`\xi\in\Gamma`, where :math:`\mathbb{Q}=\bigotimes_{i=1}^p \mathbb{Q}_i` and :math:`\Gamma=\bigotimes_{i=1}^p \Gamma_i`. 
Note that :math:`\mathbb{Q}_i` is the admissible space of the i-th parameter which can be mappd onto :math:`\Gamma_i` based on the gPCE rules, see [Xiu02]_, [Eldred09]_.
For the details of the non-intrusive PCE method refer to :ref:`gPCE-sect`.



Lagrange Interpolation
----------------------

Theory
~~~~~~
As another form of SC-based surrogates, Lagrange interpolation can be considered:

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=1}^n \hat{f}_k(\chi,\mathbf{q}) L_k(\mathbf{q}) \,,

where :math:`\hat{f}_k(\chi,\mathbf{q})=f(\chi,\mathbf{q}^{(k)})=r^{(k)}` are the training model outputs.
If the :math:`n_i` samples taken from the :math:`i`-th parameter space are represented by :math:`Q_{n_i}`, then the use of tensor-product leads to the nodal set :math:`Q_n` of size :math:`n=\prod_{i=1}^p n_i`, where,

.. math::
   Q_n= Q_{n_1} \bigotimes Q_{n_2}\bigotimes \ldots \bigotimes Q_{n_p} \,.

Correspondingly, the Lagrange bases :math:`L_k(\mathbf{q})` are constructed using the tensor-product of the bases in each of the parameter spaces: 

.. math::
   L_k(\mathbf{q})=L_{k_1}(q_1) \bigotimes L_{k_2}(q_2) \bigotimes \ldots \bigotimes L_{k_p}(q_p) \,,

where,

.. math::
   L_{k_i}(q_i) = \prod_{\substack{{k_i=1}\\{k_i\neq j}}}^{n_i} 
   \frac{q_i - q_i^{(k_i)}}{q_i^{(k_i)}-q_i^{(j)}} \,,\quad i=1,2,\ldots,p \,.

Note that the Lagrange basis satisfies :math:`L_{k_i}(q_i^{(j)})=\delta_{k_{i}j}`, where :math:`\delta` represents the Kronecker delta. 


Example
~~~~~~~
* For :math:`p=1` (one-dimensional parameter :math:`q`):

.. code-block:: python

    fInterp=lagInt(fNodes=fNodes,qNodes=[qNodes],qTest=[qTest]).val

* For :math:`p>1` (multi-dimensional parameter :math:`\mathbf{q}`):

.. code-block:: python

    fInterp=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTestList,liDict={'testRule':'tensorProd'}).val

Implementation
~~~~~~~~~~~~~~
.. automodule:: lagInt
   :members:

Notebook
~~~~~~~~
Try this `LagInt notebook <../examples/lagInt.ipynb>`_ to see how to use :code:`UQit` for Lagrange interpolation over a parameter space. 



Gaussian Process Regression
---------------------------
Theory
~~~~~~
Consider the simulator :math:`f(\mathbf{q})` where :math:`\mathbf{q}\in \mathbb{Q}`. 
The observations are assumed to be generated from the following model,

.. math::
   y = f(\mathbf{q}) + \varepsilon  \,.

Since the exact simulator :math:`f(\mathbf{q})` is not known, we can put a prior on it, which is in the form of a Gaussian process, see [Rasmussen05]_, [Gramacy20]_. 
Based on the training data :math:`\mathcal{D}`, the posterior of the :math:`f(q)`, denoted by :math:`\tilde{f}(\mathbf{q})`, is inferred. 
Without loss of generality we assume :math:`\varepsilon\sim\mathcal{N}(0,\sigma^2)`. 
Contrary to the common use of Gaussian process regression (GPR) where :math:`\sigma` is assumed to be fixed for all observations (homoscedastic noise), we are interested in cases where :math:`\sigma` is observation-dependent (heteroscedastic noise).
In the latter, we need to have a Gaussian process to infer the noise parameters, see [Goldberg98]_.
Eventually, the posteriors of :math:`\tilde{f}(\mathbf{q})` and response :math:`y` can be sampled over the parameter space, see [Rezaeiravesh20]_ and the references therein for the details. 


Example
~~~~~~~
Given the training data including the observational noise, A GPR is constructed in :code:`UQit` as,

.. code-block:: python

   gpr_=gpr(xTrain,yTrain[:,None],noiseSdev,xTest,gprOpts)
   post_f=gpr_.post_f
   post_obs=gpr_.post_y


Implementation
~~~~~~~~~~~~~~
In :code:`UQit`, the GPR is implemented using the existing Python library :code:`GPyTorch` [Gardner18]_. 
The user can similarly use any other available library for GPR as long as the code structure is kept consistent with :code:`UQit`. 

.. automodule:: gpr_torch
   :members:



Notebook
~~~~~~~~
Try `GPR notebook <../examples/gpr.ipynb>`_ to see how to use :code:`UQit` for Gaussian process regression over a parameter space. 

