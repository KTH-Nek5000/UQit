=======================
Overview \& Terminology 
=======================

.. figure:: ../_static/uqFrame.png
   :scale: 50%
   :align: center
   :alt: Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

   Schematic of different UQ problems, taken from [Rezaeiravesh18]_.

A computational model or code depends on different types of inputs and parameters.
According to [Santner03]_, the parameters can be categorized as controlled, environmental and uncertain. 
The focus of the uncertainty quantification (UQ) techniques is mainly on the last two. 


blackbox, non-intrusive

In the non-intrusive PCE, where there are a limited number of training data :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}`, there are two more main steps to construct the above expansion.

Consider uncertain parameters :math:`\mathbf{q}\in \mathbb{Q}\subset \mathbb{R}^p`.
To construct the surrogates, a limited number of samples are taken from the parameter space.
These samples are represented :math:`\{\mathbf{q}^{(i)}\}_{i=1}^n`.
In practice, the trade-off between the accuaracy of the surrogate and the cost of running the simulator determines :math:`n`.
Running the simulator at the :math:`n` parameter samples, realizations for the model outputs or QoIs, :math:`\{r^{(n)}\}_{i=1}^n`, are obtained.
Note that, the siumulator is being seen as a blackbox.
Therefore, the training data to construct the surrogate are :math:`\mathcal{D}=\{(\mathbf{q}^{(i)},r^{(i)})\}_{i=1}^n`.




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

