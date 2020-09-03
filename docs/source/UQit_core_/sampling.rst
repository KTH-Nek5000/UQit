==================
Sampling and Nodes
==================

Sampling
--------

In :code:`UQit`, different types of samples can be taken from the parameter space. 
In the big picture, the parameter samples are divided into training and test. 

To construct a surrogate or perform a UQ forward problem, we need to take training samples from :math:`\Gamma` and then we map them to :math:`\mathbb{Q}`.
In contrast the test samples which are, for instance, used to evalute the surrogates, are taken from :math:`\mathbb{Q}` and then are mapped to :math:`\Gamma`.



Implementation
~~~~~~~~~~~~~~

.. automodule:: sampling
   :members:

Example
~~~~~~~



Nodes
-----

Implementation
~~~~~~~~~~~~~~
.. automodule:: nodes
   :members:
