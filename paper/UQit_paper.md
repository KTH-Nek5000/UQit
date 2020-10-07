---
title: 'UQit: A Python package for uncertainty quantification (UQ) in computational fluid dynamics (CFD)'
tags:
  - Python
  - uncertainty quantification (UQ)
  - computational fluid dynamics (CFD)
authors:
  - name: Saleh Rezaeiravesh^[Corresponding author]
    orcid: 0000-0002-9610-9910
    affiliation: 1,2 
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
---


# Introduction

In computational physics, mathematical models are numerically solved and as a result, realizations for the quantities of interest (QoIs) are obtained. 
Even when adopting the most accurate numerical methods for deterministic mathematical models, the QoIs can still be, up to some extent, uncertain. 
Uncertainty is defined as the lack of certainty and it originates from the lack, impropriety or insufficiency of knowledge and information [@Smith:2013,@Ghanem:2017].
It is important to note that uncertainty is different from error which is attributed to the deviation of a response from a reference true value. 
In computational models, various sources of uncertainties may exist,
These include but not limited to the fidelity of the mathematical model (i.e. the extent by which the model can reflect the truth), the parameters in the models, initial data and boundary condition, finite sampling time when computing the time-averaged QoIs, the way numerical errors interact and evolve, computer arithmetic, coding bugs, geometrical uncertainties, etc. 
Various mathematical and statistical techniques gathered under the umbrella of uncertainty quantification (UQ) can be exploited to assess the uncertainty in different models and associated QoIs, see [@Smith:2013,@Ghanem:2017]. 
The UQ techniques not only facilitate systematic evaluation of validation and verification metrics, but also play a vital role in evaluation of the confidence and reliability of the data acquired in experiments and computations. 
Note that accurate accounting for such confidence intervals is crucial in data-driven engineering designs. 


In general, uncertainties can be divided into two main categories, aleatoric and epistemic [@Smith:2013]. 
The aleatoric uncertainties are random, inherent in the models, and hence, cannot be removed.
In contrast, epistemic uncertainties originate from using simplified models, insufficient data, etc.
Therefore,  they can be reduced through improving the models, for instance. 
As a general strategy in UQ, we try to reformulate the epistemic uncertainties in terms of aleatoric uncertainties so that the probabilistic approaches could be applied. 
To perform this, we can adopt the non-intrusive point of view which does not need the computational codes, hereafter simulators, to be modified.
As a result, the UQ techniques can be combined with the features in computer experiments [@Santner:2003].


These constitute the foundations of developing `UQit`, a Python package for uncertainty quantification in computational physics, in general, and computational fluid dynamics (CFD), in particular. 
In CFD, the Navier-Stokes equations are numerically solved to model the fluid flows. 
The flows are, in general, three-dimensional and time-dependent (unsteady) and at most of the Reynolds numbers relevant to the practical applications, are turbulent. 
A wide range of approaches have been used for numerical modeling of turbulence, see [Sagaut:2013].
Moving from low- toward high-fidelity approaches, the nature of the uncertainties change from modeling-dominant to numerical-driven. 
Nevertheless, the QoIs computed in the simulations can be uncertain!  


# Statement of need \& Design

Performing different types of UQ analyses in CFD is so important that it has been considered as one of the required technologies in the NASA CFD vision 2030 [@Slotnick:2014].
In this regard, `UQit` can be seen as a good match noting that it can be (one of) the first Python-based open-source packages for UQ in CFD.
In fact, there are many similarities as well as connections between UQ and the techniques in the fields of machine learning and data science in which Python libraries are rich. 
These besides the felxible design of `UQit` provide a good potential for further development of `UQit` in response to different needs. 
Due to the non-intrusive nature of the implemented UQ techniques, `UQit` treats the CFD simulator as a blackbox, therefore it can be linked to any CFD simulator conditioned on having appropriate interface.
As a possible future development, Python VTK interface can be considered for the purpose of in-situ UQ analyses which will be suitable for large-scale simulations of fluid flows on supercomputers.
The documentation for each UQ technique in `UQit` starts from providing an overview to the theoretical background and introduction of the relevent references. 
These are followed by the details of the implementation, examples, and notebooks.
The examples in each notebook are exploited not as a user guide, but also as a way to verify and validate the implementations through comparison with reference data. 
Considering these points, `UQit` provides an appropriate environment for pedagogical purposes when it comes to practical guides to UQ approaches.  


# Features

Here, a short summary of the main UQ techniques implemented in `UQit` is given. 
In general, the methods are implemented at highest required flexibility and they can be applied to any number of uncertain parameters. 
For the theoretical background, further details, and different applications in CFD, see [@Rezaeiravesh:2020].

**Surrogates** play a key role in conducting non-intrusive UQ analyses and computer experiments.
   They establish a functional between the simulator outputs (or QoIs) and the model inputs and parameters. 
   The surrogates are constructed based on a limited number of training data and once constructed, they are much less expensive to run than the simulators. 
   `UQit` uses different approaches to construct surrogates, including Lagrange interpolation, polynomial chaos expansion [@Xiu:2002,@Xiu:2007] and more importantly Gaussian process regression [@Rasmussen:2005,@Gramacy:2020]. 
   In developing `UQit`, the highest possible flexibility in constructing GPR surrogates have been considered specially when it comes to incorporating the observational uncertainties.


The goal of **uncertainty propagation** or **UQ forward problem** is to estimate how the known uncertainties in the inputs and parameters propagate into the QoIs. 
    In `UQit`, these problems are efficiently handled using non-intrusive generalized polynomial chaos expansion (PCE), see [@Xiu:2002,@Xiu:2007]. 
    For constructing a PCE, `UQit` offers a diverse set of options for the schemes for truncating the expansion, types of parameter samples, and methods to compute the coefficients in the expansion.
    For the latter, regression and projection methods can be adopted. 
    As a great feature for computationally expensive CFD simulations, compressed sensing method can be utilized when the number of training samples are less than the truncation limit. 
    
    
**Global sensitivity analysis** is performed to quantify the sensitivity of the QoIs with respect to the uncertain inputs and parameters. 
Contrary to the local sensitivity analysis, in GSA all parameters are allowed to vary simultaneously and no linearization is involved in computing sensitivities. 
In `UQit`, Sobol Sensitivity Indices (main, interaction, and total) [@Sobol:2001] are computed as indicators of GSA. 

Driven by the needs, different features can be developed and added to `UQit`.


# Acknowledgements

This work has been supported by the EXCELLERAT project which has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 823691.
Also, financial support by the Linn{\`e} FLOW Centre at KTH for SR is gratefully acknowledged.


# References
