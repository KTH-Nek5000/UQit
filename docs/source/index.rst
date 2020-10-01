.. UQit documentation master file, created by
   sphinx-quickstart on Wed Aug  5 18:11:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: ./_static/uqit_logo.pdf
   :scale: 70%
   :align: center
   :alt: A Python Toolbox for Uncertainty Quantification in CFD
   
   **A Python Toolbox for Uncertainty Quantification (UQ) in Computational Fluid Dynamics (CFD)**


:code:`UQit` is a Python toolbox for uncertainty quantification (UQ) in computational physics, in general, and in computational fluid dynamics (CFD), in particular.
Different techniques are already included to address varrious types of UQ analyses [Smith13]_, [Ghanem17]_ particularly arising in CFD.
:code:`UQit` is designed to be non-intrusively linked to CFD solvers through appropriate interfaces. 
Another important design concept in :code:`UQit` is to provide the possiblity of combining different UQ tools with each other and also with machine learning and data science techniques which can be easily added to :code:`UQit`.
Some of the main features in the current version of :code:`UQit` are listed below. 

* **Uncertainty propagation or UQ forward problem:**
Estimate how the known uncertainties in the inputs and parameters propagate into the quantities of interest (QoIs). 
This can be efficiently done by non-intrusive generalized polynomial chaos expansion (PCE), see [Xiu02]_, [Xiu07]_. 
For constructing PCE both regression and projection methods are implemented.
Using compressed sensing method PCE can be constructed using a small number of training samples.
Samples from the parameter space can be taken using different methods implemented in sampling module. 

* **Global sensitivity analysis (GSA):**
GSA is performed to quantify the sensitivity of the QoIs to the simultaneous variation of the inputs/parameters.
Contrary to local sensitivity analysis (LSA), in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities.
In :code:`UQit`, GSA is performed by Sobol sensitivity indices [Sobol01]_.

* **Surrogates:**
`UQit` uses different approaches including Lagrange interpolation, polynomial chaos expansion and more importantly GPR [Rasmussen05]_, [Gramacy20]_ technologies to construct surrogates for simulators which connect the QoIs to the inputs/parameters.
Surrogates are the pillars for conducting computer experiments [Santner03]_.
In particular, highest possible flexibility in constructing GPR surrogates have been considered when it comes to incorporating the observational uncertainties.
Combining GPR with PCE results in probabilistic PCE which is a very novel and powerful tool for CFD simulations.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   ./UQit_core_/instl_dep
   ./UQit_core_/codes_list
   ./UQit_core_/terminology
   ./UQit_core_/sampling
   ./UQit_core_/surrogate
   ./UQit_core_/uqFWD
   ./UQit_core_/gsa
   ./UQit_core_/others
   ./UQit_apps_/wrChan
   ./bib

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

