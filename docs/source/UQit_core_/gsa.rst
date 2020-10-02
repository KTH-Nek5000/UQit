=================================
Global Sensitivity Analysis (GSA)
=================================
Global sensitivity analysis (GSA) aims at quantifying the sensitivity of the model response
or quantity of interest (QoI) with respect to the variation of the uncertain parameters and inputs. 
In other words, the influence of each of the parameters in the propagated uncertainty in the QoI is measured.
In contrast to the local sensitivity analysis, in GSA all the parameters are allowed to vary simultaneously over their admissible space, [Smith13]_. 
In :code:`UQit`, the Sobol sensitivity indices [Sobol01]_ are computed to measure GSA. 

.. _sobol-sect:

Sobol Sensitivity Indices
-------------------------

Theory
~~~~~~
The following short description has been taken from [Rezaeiravesh20]_. 
For more in depth review, the reader is referred to [Sobol01]_, Chapter 15 in [Smith13]_, and [Ghanem17]_.

By analysis of variance (`ANOVA <https://en.wikipedia.org/wiki/Analysis_of_variance>`_) or Sobol decomposition [Sobol01]_, a model function is decomposed as,

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
The main Sobol indices are eventually defined as the contribution of each of :math:`D_i`, :math:`D_{ij}`, ... in the total variance :math:`D`:

.. math::
   \begin{equation}
   S_i=D_i/D\,,\quad
   S_{ij}=D_{ij}/D \,,\, \ldots \,, \quad i,j=1,2,\cdots,p
   \label{eq:sobol} \tag{2}
   \end{equation}

Example
~~~~~~~
Given samples :code:`q` with associated :code:`pdf` and response values :code:`f`, the Sobol indices in :code:`UQit` are computed by, 

.. code-block:: python

   sobol_=sobol(q,f,pdf)
   Si=sobol_.Si     #1st-order main indices
   STi=sobol_.STi   #1st-order total indices
   Sij=sobol_.Sij   #2nd-order interactions
   SijName=sobol_.SijName  #Names of Sij

Implementation
~~~~~~~~~~~~~~

.. automodule:: sobol
   :members:

Notebook
~~~~~~~~
Try this `GSA Notebook <../examples/sobol.ipynb>`_ to see how to use :code:`UQit` to compute Sobol indices. The provided examples can also be seen as a way to validate the implementation of the methods in :code:`UQit`.  

