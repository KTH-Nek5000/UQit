![`UQit`](./docsrc/source/_static/uqit_logo.png?style=centerme)

## A Python Package for Uncertainty Quantification (UQ) in Computational Fluid Dynamics (CFD)
SimEx/FLOW, Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden <br/>
#

### Features:
* **Sampling**:
  - Various stochastic and spectral types of samples

* **Uncertainty propagation or UQ forward problem**: 
  - generalized Polynomial Chaos Expansion (gPCE)
  - Probabilistic PCE (PPCE)

* **Global sensitivity analysis (GSA)**:
  - Sobol sensitivity indices

* **Surrogates**:
  - Lagrange interpolation
  - gPCE
  - Gaussian process regression (GPR) 

### Installation:
`pip install UQit`

### Documentation:
The html documentation is on [this GitHub page](https://kth-nek5000.github.io/UQit/)

### How to run a test:
The tests associated to different UQ techniques are provided in `./tests/`. 
Most of the tests have also been used in the notebooks and documentation. 
For instance, to run a test from `pce_tests.py` in `./tests/`, do the following:

`python3 -c 'import pce_tests as X;X.pce_1d_test()'`

### Required libraries:
 * General  
   - [`numpy`](https://numpy.org/)
   - [`scipy`](https://www.scipy.org/)
   - [`matplotlib`](https://matplotlib.org/)
 * Optional
   - [`cvxpy`](https://www.cvxpy.org/) (for compressed sensing in PCE)
   - [`PyTorch`](https://pytorch.org/) (for GPR)
   - [`GPyTorch`](https://gpytorch.ai/) (for GPR)

### Bugs/Questions
* In case there is a bug, please feel free to open an issue on Github. 

* Qestions/comments:
  - Saleh Rezaeiravesh, salehr@kth.se <br/>
  - Ricardo Vinuesa, rvinuesa@mech.kth.se 
  - Philipp Schlatter, pschlatt@mech.kth.se <br/>

### Publications related to UQit:
* [Rezaeiravesh S., Vinuesa R., Schlatter P., An Uncertainty-Quantification Framework for Assessing Accuracy, Sensitivity, and Robustness in Computational Fluid Dynamics, arXiv:2007.07071, 2020.](https://arxiv.org/abs/2007.07071)

## Release Notes
### Release v 1.0.2, 27.10.2020
Source code, documentation, tests and notebooks are provided for the above-listed features. 

