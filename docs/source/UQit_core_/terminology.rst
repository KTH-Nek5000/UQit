=======================
Overview \& Terminology 
=======================

.. figure:: ../_static/uqFrame.png
   :scale: 40%
   :align: center
   :alt: Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

   Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

.. _overview-sect:  

Overview
--------
A computational model or code depends on different types of inputs and parameters which according to [Santner03]_ can be categorized as controlled, environmental and uncertain. 
The focus of the uncertainty quantification (UQ) techniques is mainly on the last two. 

:code:`UQit` is designed mainly based on the needs for UQ in the CFD (computational fluid dynamics) community. 
Its connection with the CFD solvers is non-intrusive where the CFD code is treated as a blackbox. 
As a result, we always deal with discrete data which comprises of parameter samples and corresponding responses acquired by running the simulator. 
A good overview over different UQ approaches can be found for instance in [Smith13]_ and [Ghanem17]_.
Moreover, the terminology and some of materials in this documentation are taken from [Rezaeiravesh20]_.
Below, we list some of the main features of :code:`UQit`.


* **Uncertainty propagation or UQ forward problem:**

Estimates how the known uncertainties in the inputs and parameters propagate into the quantities of interest (QoIs).
These problems can be efficiently handled using non-intrusive generalized polynomial chaos expansion (PCE), see [Xiu02]_, [Xiu07]_.
In :code:`UQit`, for constructing PCE both regression and projection methods are implemented.
Using compressed sensing method, PCE can be constructed using a small number of training samples.
Samples from the parameter space can be taken using different methods implemented in :ref:`sampling_sect` module.
See the details in :ref:`uqFwd-sect`.

* **Global sensitivity analysis (GSA):**


GSA is performed to quantify the sensitivity of the QoIs to the simultaneous variation of the inputs/parameters.
Contrary to local sensitivity analysis (LSA), in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities.
In :code:`UQit`, GSA is performed by :ref:`sobol-sect` [Sobol01]_.

* **Surrogates:**

:code:`UQit` uses different approaches including Lagrange interpolation, polynomial chaos expansion and more importantly Gaussian process regression [Rasmussen05]_, [Gramacy20]_ to construct :ref:`surrogates-sect` which connect the QoIs to the inputs/parameters.
Surrogates are the pillars for conducting computer experiments [Santner03]_.
In particular, highest possible flexibility in constructing GPR surrogates have been considered when it comes to incorporating the observational uncertainties.



Nomenclature
------------
Throughout this documentation, we adopt the terminologies and nomenclature from [Rezaeiravesh20]_, as summarized in the following table. 

======================== =============================================
      **Symbol**                       **Definition**
------------------------ ---------------------------------------------
QoI                      Quantity of Interest
:math:`f(\cdot)`         Model function or simulator
:math:`\tilde{f}(\cdot)` Surrogate
:math:`\chi`             Controlled parameter
:math:`q_i`              i-th uncertain parameter (single-variate)
:math:`\mathbf{q}`       Multivariate uncertain parameter
:math:`\mathbf{q}^{(j)}` j-th sample of :math:`\mathbf{q}`
:math:`p`                Dimension of :math:`\mathbf{q}`
:math:`\mathbb{Q}`       Admissible space of :math:`\mathbf{q}`
:math:`\mathbb{Q}_i`     Admissible space of :math:`q_i`
:math:`r`                Model response, output or QoI
:math:`\bigotimes`       Tensor product
:math:`\mathcal{U}`      Uniform distribution
:math:`\mathcal{N}`      Normal (Gaussian) distribution
======================== =============================================

