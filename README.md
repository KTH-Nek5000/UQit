![`UQit`](./docs/source/_static/UQit.png?raw=true "UQit, a Python package for UQ in CFD")
## `UQit`, A Python Toolbox for Uncertainty Quantification in CFD
Saleh Rezaeiravesh, salehr@kth.se <br/>
SimEx/FLOW, Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden
#
#
### Add path to ~/.bashrc
  UQit=<path-on-the-disk/>
  source ~/.bashrc

### Required libraries:
 - [`numpy`](https://numpy.org/)
 - [`scipy`](https://www.scipy.org/)
 - [`matplotlib`](https://matplotlib.org/)
 - [`cvxpy`](https://www.cvxpy.org/) (convex optimization used in /general/linAlg.py/myLinearRegress()) <>
   * Install via `pip` inside `conda`:
   * `conda install pip`
   * `pip install cvxpy`
 - [`PyTorch`](https://pytorch.org/)
 - [`GPyTorch`](https://gpytorch.ai/)

## Release Notes
### UQit-20.10.1



# Overview to the `UQit` modules:
* **Data extraction**: 
  `UQit` is designed to be non-intrusively linked to any CFD solver provided having appropriate interfaces. 
  The interface can be either coded or be based on VTK, for instance. 

* **Uncertainty propagation or UQ forward problem**: 
  Estimate how the known uncertainties in the inputs and parameters propagate into the quantities of interest (QoIs). 
  This is done by non-intrusive polynomial chaos expansion (PCE). For constructing PCE both regression and projection methods are implemented. 
  For the regression we also have compressed sensing available which makes it possible to include very small number of training samples. 
  We have a separate module with different techniques for sampling from the space of the inputs/parameters.

* **Global sensitivity analysis (GSA)**: is performed to quantify the sensitivity of the QoIs to the simultaneous variation of the inputs/parameters. 
  Contrary to the local sensitivity analysis (LSA), in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities (Sobol indices). 
  Consequently the GSA results are much more informative than LSA indicators. 

* **Gaussian process regression (GPR)**: 
  `UQit` uses GPR technology to construct surrogates for the QoIs in the space of the inputs/parameters. 
  In our implementation, we have considered the highest possible flexibility in constructing GPR surrogates when it comes to incorporating the observational uncertainties. 
  We have both homoscedastic and heteroscedastic noise structures which the latter allows for observation-dependent uncertainty. 
  Combining GPR with PCE results in probabilistic PCE which is a very novel and powerful tool for CFD simulations.

* **Time-series analysis (`UQit-ts`)**: 
  Samples from the flow fields are highly correlated in both space and time. 
  This makes the problem of analyzing them very challenging. 
  In particular, we are interested in computing the uncertainty in the sample mean (time-average) of the time samples of the flow quantities taken at some user-defined spatial locations inside the domain. 
  For this purpose there are several methods (estimators) implemented in module `UQit-ts`. 
  These include batch-based methods, autocorrelation-based methods, and autoregressive models. 
  We have been studying the computational cost and accuracy of these different estimators. 
  Furthermore, we have succeeded to use Uqit-ts with VTK interface by which samples from a running Nek5000 case are extracted. 
