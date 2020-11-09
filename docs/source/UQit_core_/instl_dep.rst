=============================
Installation and Dependencies
=============================


Installation
------------
:code:`UQit` can be found on `PyPI`, see `here <https://pypi.org/project/UQit/>`_. 

To install :code:`UQit`:

.. code-block::

   pip install UQit

To build the documentation:   

First, you need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ to be installed:

.. code-block::

  conda install sphinx
  conda install -c conda-forge nbsphinx

Then in `/docs`,

.. code-block::

   make html
   


Dependencies
------------



* Required:

  - `numpy <https://numpy.org/>`_
  - `scipy <https://www.scipy.org/>`_
  - `matplotlib <https://matplotlib.org/>`_


* Optional:

  - `cvxpy <https://www.cvxpy.org/>`_ (for compressed sensing in PCE)
  - `GPyTorch <https://gpytorch.ai/>`_ (for GPR)
  - `PyTorch <https://pytorch.org/>`_ (for GPR)




