% Function comparing the different evaluation model criteria (AIC, BIC, Cp,
% loss estimators, etc for a given exploration and a given estimator in the
% linear regression model
%
%       y = X*beta + epsilon
%
% where y is the variable we wish to explain by the other variables in the
% (fixed) design matrix X, beta is the unknown (sparse) regression coefficient 
% we wish to estimate and epsilon is the noise vector.
%
% Input
% -----
% - explor  = string with possible value:
%       o "subset" for all subsets (or exhaustive) exploration
%       o "forward" for forward selection
%       o "backward" for backward elimination
%       o "lasso" for least absolution shrinkage and selection operator (cf
%       Tibshirani, 1996)
%       o "mcp" or "firm" for Firm Shrinkage (Bruce & Gao, 1995) and its
%       extension Minimax Concave Penalty (Zhang, 2010)
%       o "garrote" for non-negative garrote estimator (Breiman, 1995)
%       o "adalasso" or "adaptive" for Adaptive Lasso (Zou, 2006)
%       o "enet" for Elastic Net (Zou, 2006)
%       o "adanet" for Adaptive Elastic Net (Zou, 2009)
%       o "wide_path" for a randomization of lasso's regularization path
% - estim   = string with possible value:
%       o "ls" for Least-Squares Estimator
%       o "js" for James-Stein estimator (James & Stein, 1961)
%       o "gjs" for generalized James-Stein estimator
%       o "ridge" for Ridge regression (Hoerl & Kennard, 1971)
%       o "lasso" for least absolution shrinkage and selection operator (cf
%       Tibshirani, 1996)
%       o "mcp" or "firm" for Firm Shrinkage (Bruce & Gao, 1995) and its
%       extension Minimax Concave Penalty (Zhang, 2010)
%       o "garrote" for non-negative garrote estimator (Breiman, 1995)
%       o "adalasso" or "adaptive" for Adaptive Lasso (Zou, 2006)
%       o "enet" for Elastic Net (Zou, 2006)
%       o "adanet" for Adaptive Elastic Net (Zou, 2009)
% - distrib = string for the name of the distribution of the noise vector epsilon.
%       Possible values:
%       o "unif" for uniform distribution on an interval
%       o "unifSS" for uniform distribution on a sphere
%       o "gauss" for Gaussian distribution
%       o "t_mg" for multivariate Student distribution
%       o "kotz" for multivariate Kotz distribution
% - exemple = string for the name of the example. Possible values:
%       o "martin" for the example in Fourdrinier & Wells, 1994
%       o "tibshirani1", "tibshirani2", "tibshirani3" or "tibshirani4" for 
%           Example 1 to 4 in Tibshirani, 1996
%       o "breiman" for Example 2 in Breiman, 1996
%       o "zou0" for Model 0 in Zou, 2006 (adaptive lasso) and "zou1" for Model 1 in Zou,
%       2006 (elastic net)
% - options = structure of data with the following elements
%       o options.orthog : 0 if general design, 1 if orthoginal design
%       (faster algorithm in the latter case)
%       o options.nb_var : number of "replicates" of the randomized
%       regularization path in wide_path
%       o options.firm / options.mcp : value of second hyperparamter for
%       Firm Shrinkage / Minimax concave penalty. Default value = 2
%       o options.ada : value of second hyperparamter for
%       Adaptive Lasso / Garrote / Adaptive Elastic net. Default values = 2
%       / 1 / 2.
%       o options.enet : value of second hyperparamter for
%       Elastic net. Default value =  0.3 
%
% A. Boisbunon, 02/2013



function [beta_best,beta_chap,crit,df] = ModSel(X,y,explor,estim,evalcrit,options)

if ((nargin<6)||~isfield(options,'orthog'))
    options.orthog=0;   % orthog=1 --> X is an orthogonal matrix 
                        % orthog=0 --> X is general
end



%       --------------------------------------
%       ESTIMATION OF BETA FOR VARIOUS SUBSETS
%       --------------------------------------

[beta_chap, ls, df] = ExplorEstim(X, y, explor, estim, options) ;


%       ------------------------------------------
%       COMPUTE THE GENERALIZED DEGREES OF FREEDOM
%       ------------------------------------------

if isempty(df)
    h=10^(-4); % precision for computing the degrees of freedom
    df = dof(X,y, beta_chap,h, explor, estim, options);
end

%       ---------------------------
%       EVALUATION OF THE SOLUTIONS
%       ---------------------------


if nargin<6
    crit = EvalModel(X,y,beta_chap,df,ls,evalcrit,explor, estim,options);
else
    if (~isfield(options,'var_estim')&&~isfield(options,'full'))
        crit = EvalModel(X,y,beta_chap,df,ls,evalcrit,explor, estim,options);
    else
        crit = EvalModel(X,y,beta_chap,df,ls,evalcrit,explor, estim,options,options.var_estim,options.full);
    end
end

%       -------------------------------------------------------
%       FIND THE BEST SUBSET FOR EACH CRITERION AND COMPUTE THE
%                   CORRESPONDING PREDICTION ERROR
%       -------------------------------------------------------


[val,indmin] = min(crit);
beta_best = beta_chap(:,indmin);

end
