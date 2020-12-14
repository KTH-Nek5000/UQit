================
Other Core Codes
================

Analytical Test Functions
-------------------------
Analytical model functions to test and validate implementation of different UQ techniques.

.. automodule:: analyticTestFuncs
   :members:


Surrogate to Surrogate
----------------------
Interpolate values from one surrogate to another surrogate.

.. automodule:: surr2surr
   :members:


Statistical Tools
-----------------

.. automodule:: stats
   :members:


Linear Algebra
--------------
Tools for linear algebra.

To solve a linear system which is under-determined, the compressed sensing method is used. 
The required optimization is handled by :code:`cxvpy` [Diamond16]_. 
Different solvers can be used for this purpose, a list of which can be obtained by 
`cvxpy.installed_solvers()`. 
The required options for each solver can be found in `this cvxpy page <https://www.cvxpy.org/tutorial/advanced/index.html?highlight=installed_solvers>`_.
Note that the default solver is directly specified in :code:`linAlg.myLinearRegress()`.


.. automodule:: linAlg
   :members:

Reshaping Tools
---------------
Tools for converting and reshaping arrays and lists.

.. automodule:: reshaper
   :members:

Tools for Printing and Writing
------------------------------
Tools for printing or writing data in file

.. automodule:: write
   :members:

