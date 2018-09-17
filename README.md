mst
===

Model Selection Toolbox for Matlab/Octave


# Introduction

List of features:

- Linear estimators: 
	- Lasso, 
	- Adaptive Lasso, 
	- Elastic net, 
	- Adaptive elastic net, 
	- MCP
	- Ridge regression estimator
	- Restricted least-squares estimator
	- (Generalized) James-Stein estimator
- Model selection: SURE, AIC, Cp, BIC, AICc, AIC3, SRM, Cross validation, Slope Heuristics, loss estimators
- Random generation of multivariate spherically symmetric distributions: Gaussian, Student, Kotz, Laplace, Bessel, Power exponential, etc.

# Install

# Documentation

This toolbox works for sparse linear models of the type y = X*beta + epsilon, where beta is the regression vector to be estimated and is assumed to be sparse (non-zero coefficients corresponding to explanatory variables).
In this setting, we use the quadratic loss || beta_chap - beta ||^2 as a benchmark for comparing different models. 
- 'ex_ModSel.m' is an example of how the code should be called.
- 'ModSel.m' is the main function for the comparison of several models and the selection of the best one.
- 'ExplorEstim.m' uses one method (stepwise or regularization) to determine the order in which the variables should be explored, and then either keeps the corresponding estimator or changes for another one (among least-squares, ridge regression, James-Stein estimator, generalized James-Stein estimator) computed on the resulting subset.
- 'Shrinkage.m' considers the special case where the design matrix X is orthogonal and thus uses a more efficient way to compute the regularization-type estimators.
- 'monLAR.m' finds the Lasso regularization path and computes both the Lasso estimate for beta and the least-squares estimate on the variables of X selected by Lasso.
- 'monMCP.m' finds the MCP regularization path and computes both the MCP estimate for beta and the least-squares estimate on the variables of X selected by MCP.
- 'stepwise_path.m' computes the Forward Regression and Backward Elimination.
- 'dof.m' computes the estimated degrees of freedom through directional derivative (when no analytical form is given).
- 'EvalModel.m' computes a criteria for evaluating different models.
- 'estimCouts.m' computes the unbiased and corrective loss estimates.
- 'genDataLiterature.m' generates examples (i.e. X and beta) taken from several articles in the literature.
- 'randSS.m' generates random variables from multivariate spherically symmetric distributions.
- 'ex_randSS.m' is an example of how to use the random generator.
- 'canonical.m' transforms the usual linear model in the canonical form.

