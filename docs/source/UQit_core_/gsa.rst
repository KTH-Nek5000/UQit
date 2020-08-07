=================================
Global Sensitivity Analysis (GSA)
=================================
Global sensitivity analysis (GSA) aims at quantifying the sensitivity of a model response
or quantity of interest (QoI) with respect to the variation of the uncertain parameters and inputs. 
In other words, the influence of each of the parameters in the propagated uncertainty in the QoI is measured. 
It is trivial that the dimensionality of the parameters should be greater than one, i.e. :math:`p>1`.
In contrast to the local sensitivity analysis, in GSA all the parameters are allowed to vary simeltanouesly over their own admissible space. 


Sobol Sensitivity Indices
-------------------------

Theory
~~~~~~
By analysis of variance (`ANOVA <https://en.wikipedia.org/wiki/Analysis_of_variance>`_) or `Sobol decomposition <https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706>`_, a model function is decomposed as,

.. math::
  f(\mathbf{q}) =
  f_0 +\sum_{i=1}^p f_i(q_i) + \sum_{1\leq i<j\leq p} f_{ij}(q_i,q_j)+\cdots\,,
  \label{eq:anova_f}\tag{1}

where, :math:`f_0` is the mean of :math:`f(\mathbf{q})`, :math:`f_i(q_i)` specify the contribution of each parameter, :math:`f_{ij}(q_i,q_j)` denote effects of interaction between each pair of parameters, and so on for other interactions.
These contributors are defined as,

.. math::
   \begin{eqnarray*}
   f_0 &=& \mathbb{E}_\mathbf{q}[f(\mathbf{q})] \,, \\
   f_i(q_i) &=&\mathbb{E}_\mathbf{q}[f(\mathbf{q})|q_i] - f_0 \,, \\
   f_{ij}(q_{i},q_j) &=& \mathbb{E}_\mathbf{q}[f(\mathbf{q})|q_i,q_j] -f_i(q_i) -f_j(q_j) - f_0 \,.
   \end{eqnarray*}

Here, :math:`\mathbb{E}_\mathbf{q}[f(\mathbf{q})|q_i]`, for instance, denotes the expected value of :math:`f(\mathbf{q})` conditioned on fixed values of :math:`q_i`.
Similar to Eq. \eqref{eq:anova_f}, the total variance of :math:`f(\mathbf{q})`, denoted by :math:`D`, is decomposed as,

.. math::
   \begin{equation}
   \mathbb{V}_\mathbf{q}[f(\mathbf{q})] = D=\sum_{i=1}^p D_i + \sum_{1\leq i<j\leq p} D_{ij} + \cdots \,,
   \end{equation}

where, :math:`D_i=\mathbb{V}_\mathbf{q}[f_i(q_i)]`, :math:`D_{ij}=\mathbb{V}_\mathbf{q}[f_{ij}(q_i,q_j)]`, and so on.
The main `Sobol indices <https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706>`_ are eventually defined as the contribution of each of :math:`D_i`, :math:`D_{ij}`, ... in the total variance :math:`D`:

.. math::
   \begin{equation}
   S_i=D_i/D\,,\quad
   S_{ij}=D_{ij}/D \,,\, \ldots \,, \quad i,j=1,2,\cdots,p
   \label{eq:sobol} \tag{2}
   \end{equation}

This short description has been taken from `Rezaeiravesh et al. <https://arxiv.org/abs/2007.07071>`_. 
For more in depth review, the reader is referred to `Sobol <https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706>`_, `Smith, Chapter 15 <https://rsmith.math.ncsu.edu/UQ_TIA/>`_, and `UQ Handbook <https://www.springer.com/gp/book/9783319123844>`_.


Implementation
~~~~~~~~~~~~~~
In :code:`UQit`, the Sobol sensitivity indices [sobol01]_ are computed to measure GSA. 
These indices are derived based on `variance decomposition <https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_. 
The coding for computing the Sobol indices is implemented in :code:`sobol.py`. 
The `ANOVA (analysis of variance) or Sobol decomposition <https://en.wikipedia.org/wiki/Analysis_of_variance>`_ of the model response :math:`f(\chi,\mathbf{q})` is performed by :code:`sobolDecomposCoefs(Q,f)`. 
Then, the Sobol indices for different orders of interaction between the parameters are computed by :code:`sobol(Q,f)`.

Here, :code:`Q` is a list of :math:`p` 1D :code:`numpy` arrays :math:`Q_1, Q_2, \cdots,Q_p`, where :math:`Q_i` contains the samples at the :math:`i`-th dimension in the parameter space. 

.. tip::
   Try this `notebook`_ to see how to use :code:`UQit` to compute Sobol indices. The provided examples can also be seen as a way to validate the implementation of the methods in :code:`UQit`.  

.. [sobol01] `Sobol, I. Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates. Mathematics and Computers in Simulation, 55(1):271 – 280, 2001. <https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706>`_

.. _notebook: ../examples/sobol.ipynb
