% Function computing the generalized degrees of freedom of an estimator by
% directional derivative.
%
% The generalized degrees of freedom are given by the formula
%
%   df(beta_chap) = div_y(X*beta_chap),
%
% where div_y is the divergence (sum of partial derivatives) with respect
% to y.
%
% Input (explor, estim and options are the same as for ExplorEstim.m)
% -----
%   - X       = matrix of size n*p
%   - y       = vector of size n*1
%   - explor  = string with one of the following values : 
%       o 'subset' for all subset/exhaustive exploration), 
%       o 'forward' for forward selection,
%       o 'backward' for backward elimination
%       o 'lasso' (cf Tibshirani),
%       o 'mcp' for minimax concave penalty (cf Zhang),
%       o 'adaptive' or 'adalasso' for adaptive lasso (cf Zou),
%       o 'garrote' (cf Breiman),
%       o 'enet' for elastic net (cf Zou),
%       o 'adanet' for adaptive elastic net (cf Zou),
%       o 'wide_path' for randomized lasso's regularization path.
%   - h     = step for the derivation.
%   - estim   = string with one of the following values :
%       o 'ls' for the least-squares estimator, 
%       o 'js' for the James-Stein estimator, 
%       o 'gjs' for the generalized James-Stein estimator.
%       o 'lasso' (cf Tibshirani),
%       o 'mcp' for minimax concave penalty (cf Zhang),
%       o 'adaptive' or 'adalasso' for adaptive lasso (cf Zou),
%       o 'garrote' (cf Breiman),
%       o 'enet' for elastic net (cf Zou),
%       o 'adanet' for adaptive elastic net (cf Zou),
%       o 'ridge' for ridge regression.
%   - options = structure with the following elements :
%       o options.orthog = 1 if X is an orthogonal matrix (faster)
%                        = 0 otherwise.
%       o options.firm : second hyperparameter of Firm Shrinkage or MCP.
%                        Default value : 2.
%       o options.ada  : second hyperparameter of Adaptive methods.
%                        Default value : 2 (or 1 if garrote).
%       o options.enet : second hyperparameter of Elastic net.
%                        Default value : 0.2.
% 
% Output
% ------
%   - df = number of generalized degrees of freedom.
%
% A. Boisbunon, 10/2012.
%

function [df] = dof(X, y, beta_chap, h, explor, estim, options)

    n = length(y); % No. of observations
    
    deriv_dir = zeros(n,size(beta_chap,2)) ; % Initialization
    for i=1:n
        % Step 1: add the step value to one of the composant of y
        y_eps = y+h*[zeros(i-1,1);1;zeros(n-i,1)]; 
        % Step 2: compute the new estimator with the new y
        beta_chap_eps = ExplorEstim(X,y_eps, explor, estim, options) ;
        % Step 3: compute the approximate derivative by differences
        deriv_dir(i,:) = (X(i,:)*beta_chap_eps-X(i,:)*beta_chap)/h;
    end

    % Step 4: compute the sum of directional derivatives, giving the
    % generalized degrees of freedom
    df = sum(deriv_dir,1);
    df = df(:);

end