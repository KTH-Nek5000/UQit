=======================
Overview \& Terminology 
=======================

.. figure:: ../_static/uqFrame.png
   :scale: 50%
   :align: center
   :alt: Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

   Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

Overview
--------
A computational model or code depends on different types of inputs and parameters which according to [Santner03]_, can be categorized as controlled, environmental and uncertain. 
The focus of the uncertainty quantification (UQ) techniques is mainly on the last two. 

:code:`UQit` is designed mainly based on the needs for UQ in the CFD community. 
Its connection with the CFD solvers is non-intrusive where the CFD code is treated as a blacbox. 
As a result, we always deal with discrete data which include parameter samples and asociated responses acquired by running the simulator. 
A good overview over different appraoches can be found for instance in [Smith13]_ and [Ghanem17]_.
The table below provides terminology related to this documentation and is based on [Rezaeiravesh20]_.


Nomenclature
------------

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

