.. UQit documentation master file, created by
   sphinx-quickstart on Wed Aug  5 18:11:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====
UQit 
====
A Python Toolbox for Uncertainty Quantification in CFD
------------------------------------------------------

:code:`UQit` is a Python package for uncertainty quantification (UQ) in computational physics, in general, and in computational fluid dynamics (CFD), in particular.
Different techniques are included for various type of UQ analysis.

Uncertainty propagation or UQ forward problem:
----------------------------------------------
  Estimate how the known uncertainties in the inputs and parameters propagate into the quantities of interest (QoIs).
  This is done by non-intrusive polynomial chaos expansion (PCE). For constructing PCE both regression and projection methods are implemented.
  For the regression we also have compressed sensing available which makes it possible to include very small number of training samples.
  We have a separate module with different techniques for sampling from the space of the inputs/parameters.

Global sensitivity analysis (GSA):
----------------------------------
  is performed to quantify the sensitivity of the QoIs to the simultaneous variation of the inputs/parameters.
  Contrary to the local sensitivity analysis (LSA), in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities (Sobol indices).
  Consequently the GSA results are much more informative than LSA indicators.

Gaussian process regression (GPR):
----------------------------------
  `UQit` uses GPR technology to construct surrogates for the QoIs in the space of the inputs/parameters.
  In our implementation, we have considered the highest possible flexibility in constructing GPR surrogates when it comes to incorporating the observational uncertainties.
  We have both homoscedastic and heteroscedastic noise structures which the latter allows for observation-dependent uncertainty.
  Combining GPR with PCE results in probabilistic PCE which is a very novel and powerful tool for CFD simulations.



.. code-block:: bash

   mkdir XXX


.. code-block:: python   

   import UQit
     



.. note::
   blah blah

Examples and Documentation
---------------------------



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./UQit_core_/instl_dep
   ./UQit_core_/codes_list
   ./UQit_core_/terminology
   ./UQit_core_/sampling
   ./UQit_core_/surrogate
   ./UQit_core_/uqFWD
   ./UQit_core_/gsa



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
