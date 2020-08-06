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
In :code:`UQit`, the Sobol sensitivity indices [sobol01]_ are computed to measure GSA. 
These indices are derived based on `variance decomposition <https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_. 
The coding for computing the Sobol indices is implemented in :code:`sobol.py`. 
The `ANOVA (analysis of variance) or Sobol decomposition <https://en.wikipedia.org/wiki/Analysis_of_variance>`_ of the model response :math:`f(\chi,\mathbf{q})` is performed by :code:`sobolDecomposCoefs(Q,f)`. 
Then, the Sobol indices for different orders of interaction between the parameters are computed by :code:`sobol(Q,f)`.

Here, :code:`Q` is a list of :math:`p` 1D :code:`numpy` arrays :math:`Q_1, Q_2, \cdots,Q_p`, where :math:`Q_i` contains the samples at the :math:`i`-th dimension in the parameter space. 

.. tip::
   A `notebook`_ is provided to show in detail how to use :code:`UQit` to compute Sobol indices. The examples can also be used to validate the implementation.  



.. [sobol01] `Sobol, I. Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates. Mathematics and Computers in Simulation, 55(1):271 – 280, 2001. <https://www.sciencedirect.com/science/article/abs/pii/S0378475400002706>`_

.. _notebook: ../examples/sobol.ipynb
