========================
Terminology and Overview
========================

A computational model depend on different types of inputs and parameters.
According to [Santner et al.], the parameters can be categorized as controlled, environmental and uncertain. 
The focus of the uncertainty quantification (UQ) techniques is on the 

blackbox, non-intrusive



Nomenclature
============

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

