=========
Sampling 
=========

Sampling
--------
In :code:`UQit`, different types of samples can be taken from the parameter space. 
In the big picture, the parameter samples are divided into training and test. 
To construct a surrogate or perform a UQ forward problem, we need to take training samples from :math:`\Gamma` and then we map them to :math:`\mathbb{Q}`.
In contrast the test samples which are, for instance, used to evalute the surrogates, are taken from :math:`\mathbb{Q}` and then are mapped to :math:`\Gamma`.

Available types of training samples:

 * :code:`GQ`: Gauss-Quadrature nodes

   can be used with distributions :code:`Unif`, :code:`Norm`
 * :code:`GLL`: Gauss-Lobatto-Lgendre nodes   
 * :code:`unifSpaced`: Uniformly-spaced samples   
 * :code:`unifRand`: Uniformly distributed random samples   
 * :code:`normRand`: Gaussian distributed random samples
 * :code:`Clenshaw`: Clenshaw points
 * :code:`Clenshaw-Curtis`: Clenshaw-Curtis points

Available types of test samples:

 * :code:`GLL`: Gauss-Lobatto-Lgendre nodes
 * :code:`unifSpaced`: Uniformly-spaced points
 * :code:`unifRand`: Uniformly distributed random
 * :code:`normRand`: Gaussian distributed random

Note that the argument :code:`qInfo`:
 * contains the parameter range, if the parameter is :code:`Unif`
 * contains the mean and variance, if the parameter is :code:`Norm` (:math:`\sim \mathcal{N}(\mu,\sigma^2)`)


Example
~~~~~~~

.. code-block:: python

   tr_=trainSample(sampleType=’GQ’,GQdistType=’Unif’,qInfo=[2,3],nSamp=10) 
   tr_=trainSample(sampleType=’NormRand’,qInfo=[2,3],nSamp=10) 
   tr_=trainSample(sampleType=’GLL’,qInfo=[2,3],nSamp=10)


.. code-block:: python

   ts_=testSample(sampleType=’unifRand’,GQdistType=’Unif’,qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType=’unifRand’,qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType=’normRand’,GQdistType=’Norm’,qBound=[-1,3],qInfo=[0.5,2],nSamp=10) 
   ts_=testSample(sampleType=’unifSpaced’,GQdistType=’Norm’,qBound=[-1,3],qInfo=[0.5,2],nSamp=10) 
   ts_=testSample(sampleType=’unifSpaced’,GQdistType=’Unif’,qBound=[-1,3],nSamp=10) 
   ts_=testSample(sampleType=’GLL’,qBound=[-1,3],nSamp=10)


Implementation
~~~~~~~~~~~~~~

.. automodule:: sampling
   :members: 


Implementation
~~~~~~~~~~~~~~
Some of tha sampling methods rely on generating nodes from mathematical polynomials, for instance.
.. automodule:: nodes
   :members:
