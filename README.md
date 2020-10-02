![`UQit`](./docs/source/_static/uqit_logo.png?raw=true)
## A Python Package for Uncertainty Quantification (UQ) in Computational Fluid Dynamics (CFD)
Saleh Rezaeiravesh, salehr@kth.se <br/>
SimEx/FLOW, Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden
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

### Build documentation:
First, you need [`Sphinx`](https://www.sphinx-doc.org/en/master/) to be installed: 
* `conda install sphinx`
* `conda install -c conda-forge nbsphinx`

Then,
* `cd docs`
* `make html`

Open `index.html` in `docs/build/html/`

### Required libraries:
 - [`numpy`](https://numpy.org/)
 - [`scipy`](https://www.scipy.org/)
 - [`matplotlib`](https://matplotlib.org/)
 - [`cvxpy`](https://www.cvxpy.org/) 
 - [`PyTorch`](https://pytorch.org/)
 - [`GPyTorch`](https://gpytorch.ai/)

## Release Notes
### Release 1, 10.10.2020
Source code, documentation, tests and notebooks are provided for the above-listed features. 

