mst
===

Model Selection Toolbox for Matlab/Octave


# Introduction

List of features:

- Linear estimators: 
	- Lasso, 
	- MCP, 
	- SCAD
- Model selection: Stein, AIC, Cp, BIC
- Random generation of shperically spheric distributions: Gaussian, Kotz, ...

# Install

# Documentation

This toolbox works for linear models of the type y = X*beta + epsilon, where beta is estimated via the least-squares estimate on the variables of X selected by Lasso.
The loss considered here is the quadratic loss: || beta_chap - beta ||^2.
- 'simu_example.m' is an example of how the code should be called.
- 'compCouts.m' is a comparison between the unbiased and corrective loss estimates, and also with AIC and BIC. 
- 'estimCouts.m' computes the unbiased and corrective loss estimates.
- 'canonical.m' transforms the usual linear model in the canonical form.
- 'monLAR2.m' finds the Lasso regularization path and computes both the Lasso estimate for beta and the least-squares estimate on the the variables of X selected by Lasso.
- 'randSS.m' generates random variables from spherically symmetric distributions (still needs further work for generating the radius...).
