title: 'UQit: A Python package for uncertainty quantification (UQ) in computational fluid dynamics (CFD)'
tags:
  - Python
  - uncertainty quantification (UQ)
  - computational fluid dynamics (CFD)
authors:
  - name: Saleh Rezaeiravesh^[Corresponding author]
    orcid: 0000-0002-9610-9910
    affiliation: "1,2" 
  - name: Ricardo Vinuesa
    affiliation: 1,2
  - name: Philipp Schlatter
    affiliation: 1,2
affiliations:
 - name: SimEx/FLOW, Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden
   index: 1
 - name: Swedish e-Science Research Centre (SeRC), Stockholm, Sweden
   index: 2
date: 2 October 2020
bibliography: UQit_paper.bib


# Introduction

In computational physics, a set of mathematical equations (models) are numerically solved and as a result, realizations for the quantities of interest (QoIs) are obtained. 
Even when adopting the most accurate numerical methods for deterministic models, the QoIs can still be up to some extent uncertain. 
Uncertainty is the lack of certainty and originates from the lack of or improper knowledge and information [@Smith:2013,@Ghanem:2017].
Note that uncertainty is different from error which is the measure of deviation of a response from a reference true value. 
In computational models, various sources of uncertainties may exist including the fidelity of the mathematical model (i.e. the extent the model can reflect the truth), the parameters in such models, initial data and boundary condition, finite sampling time when computing the statistics, the way numerical errors interact and evolve, computer arithmetic, coding bugs, geometrical uncertainties, etc. 
Various mathematical and statistical techniques gathered under the umbrella of uncertainty quantification (UQ) can be exploited to assess the uncertainty in different models and associate QoIs, see [@Smith:2013,@Ghanem:2017]. 
The UQ techniques not only facilitate systematic evaluation of validation and verification metrics, but also play a vital role in evaluation of confidence and reliability of data acquired in experiments and computations. 
Note that such confidence intervals are crucial in data-driven engineering designs. 

In general, uncertainties can be divided into two main categories, aleatoric and epistemic. 
The aleatoric uncertainties are random and inherent in the models and hence cannot be removed.
In contrast, epistemic uncertainties originate from using simplified model, insufficient data, ... , which can be reducible by improving the models, for instance. 
As a general approach, in UQ we try to reformulate the epistemic uncertainties in terms of aleatoric so that the probabilistic approaches can be applied. 
To perform this, we can adopt the non-intrusive point of view which does not need the computational codes, hereafter simulator, to be modified.
As a result, the UQ techniques can be combined with the features in computer experiments [@Santner:2003].


This is what has been used in developing UQit that is a Python package for uncertainty quantification in computational physics, in general, and computational fluid dynamics (CFD), in particular. 
In computational fluid dynamics (CFD), the Navier-Stokes equations or variations of them are numerically solved to model the fluid flows. 
The flows are, in general, three dimensional and time-dependent (unsteady) and at most of Reynolds numbers relevant to practical applications, are turbulent. 
Application of high-fidelity scale-resolving approaches to simulate turbulent flows require a high computational cost, while reduce the epistemic uncertainties due to turbulence modeling. 



# Statement of need \& Design

Performing different types of UQ analyses in CFD is so important that it has been considered in the NASA CFD vision 2030 [@Slotnick:2014].
UQit can be seen as a good fit to such needs, noting that it can be the only Python-based open-source package for UQ in CFD.
Due to the non-intrusive nature of the implemented UQ methods, UQit treats the CFD simulator as a blackbox, therefore it can be used with any CFD simulator conditioned on having appropriate interface provided.
As a feature development, Python VTK interface can be considered for the purpose of in-situ UQ analyses which will be suitable for large-scale simulations of fluid flows on supercomputers.
There are many similarities as well as connections between UQ and the techniques in the fields of machine learning and data science in which Python libraries are rich. 
These besides the object-oriented design provide a good potential for further development of UQit in reply to different needs. 
The documentation for each UQ technique starts from providing an overview to the theoretical background and introduction of the rel event references. 
These are followed by the details of the implementation, examples, and notebooks.
The examples in each notebook are exploited not as a user guide, but also as a way to verify and validate the implementation of the UQ techniques through comparison with reference values. 
Considering these points, UQit can be viewed as an appropriate environment for pedagogical purposes.  


# Features

Here, a short summary of the main UQ techniques implemented in UQit is given. 
In general, the methods are implemented at highest required flexibility and they can be applied to any number of uncertain parameters. 
For the theoretical background, further details, and application in CFD, see [@Rezaeiravesh:2020].

**Surrogates** play a key role in conducting non-intrusive UQ analyses and computer experiments.
   They establish a functional between the simulator outputs (or QoIs) and the model inputs and parameters. 
   The surrogates are constructed based on a limited number of training data and once constructed, they are much less expensive to evaluate than the simulators. 
   UQit uses different approaches to construct surrogates, including Lagrange interpolation, polynomial chaos expansion [@Xiu:2002,@Xiu:2007] and more importantly Gaussian process regression [@Rasmussen:2005,@Gramacy:2020]. 
   In developing UQit, the highest possible flexibility in constructing GPR surrogates have been considered when it comes to incorporating the observational uncertainties.


The goal of **uncertainty propagation** or **UQ forward problem** is to estimates how the known uncertainties in the inputs and parameters propagate into the QoIs. 
    In UQit, these problems are efficiently handled using non-intrusive generalized polynomial chaos expansion (PCE), see [@Xiu:2002,@Xiu:2007]. 
    Different options have been implemented for constructing a PCE, including different schemes for truncation of the expansion.
    Moreover, to compute the coefficients, both regression and projection methods are implemented. 
    As great feature, compressed sensing method can be adopted when the number of training samples are less than the truncation limit. 
    A great flexibility is also provided in terms of generating samples from the parameter space.
    
    
**Global sensitivity analysis** is performed to quantify the sensitivity of the QoIs to the uncertain inputs and parameters. 
Contrary to local sensitivity analysis, in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities. In UQit, Sobol Sensitivity Indices [@Sobol:2001] are computed as indicators of GSA. 


# Acknowledgements
This work has been supported by the EXCELLERAT project which has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 823691.
Also, financial support by the Linn\`e FLOW Centre at KTH for SR is gratefully acknowledged.

