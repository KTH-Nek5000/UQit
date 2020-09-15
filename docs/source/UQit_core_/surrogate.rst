==========
Surrogates
==========
A surrogate is an approaximation of the actual model function over the parameter space.
Running the surrogate is computationally inexpensive which facilitates the use of different UQ techniques.
However, the predictions by the surrogate should be accurate enough compared to the actual predictions by the simulator. 
Using an additive error model, the following relation could be considered:

.. math::
   r=f(\chi,\mathbf{q})+\mathbf{\varepsilon}\,,

where :math:`\mathbf{\varepsilon}` expresses the bias and random noise.
Our task is to construct a surrogate :math:`\tilde{f}(\chi,\mathbf{q})` for actual model function (simulator).


There are different techniqus to construct a surrogate, for instance see [smith, uqHandbook, Gramacy]. 
Some of these have been implemented in :code:`UQit` in response to the needs for CFD practices. 
Here, we provide a short overview to the theory behind these techniques and explain the code through providing examples. 


Consider uncertain parameters :math:`\mathbf{q}\in \mathbb{Q}\subset \mathbb{R}^p`. 
To construct the surrogates, a limited number of samples are taken from the parameter space. 
These samples are represented :math:`\{\mathbf{q}^{(i)}\}_{i=1}^n`.
In practice, the trade-off between the accuaracy of the surrogate and the cost of running the simulator determines :math:`n`. 
Running the simulator at the :math:`n` parameter samples, realizations for the model outputs or QoIs, :math:`\{r^{(n)}\}_{i=1}^n`, are obtained. 
Note that, the siumulator is being seen as a blackbox.
Therefore, the training data to construct the surrogate are :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}_{i=1}^n`. 


Non-intrusive Polynomial Chaos Expansion
-----------------------------------------
The stochastic collocation (SC), see [uqHandbook, Chap20] method is very well suited to the described framework. 
Since, SC relies on treating the simulator as a blackbox. 
As an option non-intrusive PCE can be considered. 

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=0}^K \hat{f}_k(\chi) \Psi_{k}(\mathbf{q})

Refer to sec??? for the details on how PCE is implemented in :code:`UQit`.



Lagrange Interpolation
----------------------

Theory
~~~~~~
As another form of SC-based surrogate, Lagrange interpolation can be considered:

.. math::
   \tilde{f}(\chi,\mathbf{q}) = \sum_{k=1}^n \hat{f}_k(\chi,\mathbf{q}) L_k(\mathbf{q}) \,,

where :math:`\hat{f}_k(\chi,\mathbf{q})=f(\chi,\mathbf{q}^{(k)})=r^{(k)}` are the training model outputs.
If :math:`n_i` samples taken from the :math:`i`-th parameter space is represented by :math:`Q_{n_i}`, then the use of tensor product leads to the nodal set :math:`Q_n` of size :math:`n=\prod_{i=1}^p n_i`, where

.. math::
   Q_n= Q_{n_1} \times Q_{n_2}\times \ldots \times Q_{n_p} \,.

Correspondingly, the Lagrange bases :math:`L_k(\mathbf{q})` are constructed using the tensor product of the basis in each of the parameter spaces: 

.. math::
   L_k(\mathbf{q})=L_{k_1}(q_1) \bigotimes L_{k_2}(q_2) \bigotimes \ldots \bigotimes L_{k_p}(q_p)

where,

.. math::
   L_{k_i}(q_i) = \prod_{\substack{{k_i=1}\\{k_i\neq j}}}^{n_i} 
   \frac{q_i - q_i^{(k)}}{q_i^{(k)}-q_i^{(j)}} \,,\quad i=1,2,\ldots,p \,.

Note that the Lagrange basis satisfies :math:`L_{k_i}(q_i^{(j)})=\delta_{k_i j}` with :math:`\delta` representing Kronecker delta. 


In :code:`UQit`, the Lagrange interpolation method is implemeneted in :code:`lagrangeInterpol.py`. 
There are three methods available:

* :code:`lagrangeBasis_singleVar(qNodes,k,Q)`
  constructs the Lagrange basis in one dimension, i.e. :math:`L_{k_i}(q_i)` evaluated at a set oftest samples. 

  - :code:`qNodes` is a 1D :code:`numpy` array containing training parameter samples. 

  - :code:`k` specfies the order of the basis.

  - :code:`Q` are the test samples, a 1D :code:`numpy` array at which the basis is evaluated. 

* :code:`lagrangeInterpol_singleVar(fNodes,qNodes,Q)`
  constructs a Lagrange interpolation for a single-variate parameter (:math:`p=1`). 

  - :code:`qNodes` is a 1D :code:`numpy` array containing training parameter samples.

  - :code:`fNodes` is a 1D :code:`numpy` array containing training model outputs.

  - :code:`Q` are the test samples, a 1D :code:`numpy` array at which the basis is evaluated. 

* :code:`lagrangeInterpol_multiVar(fNodes,qNodes,Q,method)` 
  constructs a Lagrange interpolation for a multi-variate parameter (:math:`p>1`).               

  - :code:`qNodes` is a list of :math:`p` 1D :code:`numpy` arrays each containing training parameter samples for one the single-variate parameters.

  - :code:`fNodes` is a 1D :code:`numpy` array of size :math:`n` containing training model outputs.

  - :code:`Q` is a 2D :code:`numpy` array of size :math:`m \times p` containing :math:`m` test samples over the :math:`p`-D parameter space. The constructed Lagrange interpolation is evaluated at these test samples. 

Implementation
~~~~~~~~~~~~~~
.. automodule:: lagInt
   :members:

Example
~~~~~~~
* For :math:`p=1` (one-dimensional parameter :math:`q`):

.. code-block::

    fInterp=lagInt(fNodes=fNodes,qNodes=[qNodes],qTest=[qTest]).val

* For :math:`p>1` (multi-dimensional parameter :math:`q`):

.. code-block::

    fInterp=lagInt(fNodes=fNodes,qNodes=qNodes,qTest=qTestList,liDict={'testRule':'tensorProd'}).val



Notebook
~~~~~~~~
Try this `notebook <../examples/lagrangeInterp.ipynb>`_ to see how to use :code:`UQit` for Lagrange interpolation over a parameter space. 



Gaussian Process Regression
---------------------------
Theory
~~~~~~
Consider the simulator :math:`f(q)` where :math:`q\in \mathbb{Q}`. The observations can be generated from,

.. math::
   y_i = \tilde{f}(q_i) + \varepsilon_i \,, i=1,2,... \,.

where :math:`\tilde{f}(q)` is a GP acting as a surrogate of $f(q)$ and :math:`\varepsilon` is the observatio noise. 
Without loss of generality we assume :math:`\varepsilon\sim\mathcal(0,\sigma^2)`. 
Contrary to the common use of GPR, see [Rasmuseen], where :math:`\sigma` is assumed to be fixed for all observations (homoscedastic noise), we are interested in cases where :math:`sigma` is observation dependent (heteroscedastic noise).
The latter can be read-up from [???].
At the first step, given the training data :math:`\mathcal{D}`, a GPR is constructed for :math:`\tilde{f}(q)`.
Then, the posterior and posterior predictive of :math:`\tilde{f}(q)` and response :math:`y` can be sampled over the parameter space at test samples. 

Implementation
~~~~~~~~~~~~~~
In :code:`UQit`, the GPR is implemented using the existing library :code:`GPyTorch` [????]. A user can similarly use any other available library for GPR. 

.. automodule:: gpr_torch
   :members:

Example
~~~~~~~


Notebook
~~~~~~~~
Try `gpr_notebook <../examples/gpr.ipynb>`_ to see how to use :code:`UQit` for Gaussian process regression over a parameter space. 







