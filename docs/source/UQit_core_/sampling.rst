.. _sampling_sect:

=========
Sampling 
=========

Sampling
--------
In :code:`UQit`, different types of samples can be taken from the parameter space. 
From one point of view, the parameter samples are divided into training and test. 
To construct a surrogate or perform a UQ forward problem, we need to take training samples from a mapped or standardized space :math:`\Gamma` and then map them onto the parameter admissible space :math:`\mathbb{Q}`.
In contrast, the test samples which are, for instance, used to evaluate the constructed surrogates, are taken from :math:`\mathbb{Q}` and then are mapped onto :math:`\Gamma`.

Available types of training samples:

 * :code:`GQ`: Gauss-Quadrature nodes

   can be used with distributions :code:`Unif`, :code:`Norm`
 * :code:`GLL`: Gauss-Lobatto-Legendre nodes   
 * :code:`unifSpaced`: Uniformly-spaced samples   
 * :code:`unifRand`: Uniformly distributed random samples   
 * :code:`normRand`: Gaussian distributed random samples
 * :code:`Clenshaw`: Clenshaw points
 * :code:`Clenshaw-Curtis`: Clenshaw-Curtis points

Available types of test samples:

 * :code:`GLL`: Gauss-Lobatto-Legendre nodes
 * :code:`unifSpaced`: Uniformly-spaced points
 * :code:`unifRand`: Uniformly distributed random
 * :code:`normRand`: Gaussian distributed random

Note that the argument :code:`qInfo` appearing in sampling methods:
 * :code:`qInfo=[a,b]`, if the parameter is :code:`Unif` over range :math:`[a,b]`, i.e. :math:`q\sim\mathcal{U}[a,b]`
 * :code:`qInfo=[m,s]` contains the mean :math:`m` and standard-deviation :math:`s`, if the parameter is :code:`Norm`, i.e. :math:`q\sim \mathcal{N}(m,s^2)`


Example
~~~~~~~

.. code-block:: python

   tr_=trainSample(sampleType='GQ',GQdistType='Unif',qInfo=[2,3],nSamp=10) 
   tr_=trainSample(sampleType='NormRand',qInfo=[2,3],nSamp=10) 
   tr_=trainSample(sampleType='GLL',qInfo=[2,3],nSamp=10)

.. code-block:: python

   ts_=testSample(sampleType='unifRand',GQdistType='Unif',qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType='unifRand',qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType='normRand',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10) 
   ts_=testSample(sampleType='unifSpaced',GQdistType='Norm',qBound=[-1,3],qInfo=[0.5,2],nSamp=10) 
   ts_=testSample(sampleType='unifSpaced',GQdistType='Unif',qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType='GLL',qBound=[-1,3],nSamp=10)


Implementation
~~~~~~~~~~~~~~

.. automodule:: sampling
   :members: 


Nodes
-----
Some of the sampling methods rely on generating nodes from mathematical polynomials, for instance see [Canuto87]_.
The associated methods are implemented in :code:`nodes.py`.

.. automodule:: nodes
   :members:
