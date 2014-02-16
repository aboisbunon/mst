% Function computing shrinkage estimators for the linear regression model
% with orthogonal design, 
%   
%       y = X*beta + epsilon
%
% where y is the response vector, X is the orthogonal design matrix, beta
% is the unknown regression coefficient we wish to estimate, and epsilon is
% the noise vector.
%
% Input
% -----
%   - X       = matrix of size n*p
%   - y       = vector of size n*1
%   - options = structure with the following elements :
%       o options.estim : 
%           + 'ls' for Least-Squares (or Hard Threshold),
%           + 'lasso' (cf Tibshirani),
%           + 'firm' or 'mcp' for Firm Shrinkage / Minimax Concave Penalty
%           (cf Gao and Bruce or Zhang), 
%           + 'scad' (cf Fan and Li),
%           + 'adaptive' or 'adalasso' for adaptive lasso (cf Zou),
%           + 'garrote' (cf Breiman),
%           + 'enet' for elastic net (cf Zou),
%           + 'adanet' for adaptive elastic net (cf Zou),
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
%   - beta_chap = matrix of size p*(p+1).
%   - ls        = Least-squares estimator for computing the variance.
%   - df        = vector of size (p+1)*1 of the estimated degrees of freedom
%               (useful for evaluating the solutions).
%   - hyperpar  = value of the thresholds.
%
% A. Boisbunon, 08/2012

function [beta_chap, ls, df, hyperpar] = Shrinkage(X, y, options)

p = size(X,2) ;
beta_chap = zeros(p,p+1) ;
df = zeros(p+1,1) ;

[lasso,ls,hyperpar] = estimLAR(X,y);

for i=2:1:p+1
    switch options.estim
        case 'lasso'
            beta_chap(:,i)=lasso(:,i);
            df(i) = sum((abs(lasso(:,i))>hyperpar(i))) ;
        case 'ls'
            beta_chap(:,i)=ls(:,i);
            df(i) = sum((abs(lasso(:,i))>hyperpar(i))) ;
        case {'firm','mcp'}
            if (hyperpar(i)~=0)
                beta_chap(:,i)=options.firm*lasso(:,i)/(options.firm-1).*(abs(ls(:,i))<=options.firm*hyperpar(i))+ls(:,i).*(abs(ls(:,i))>options.firm*hyperpar(i));
                df(i) = sum((abs(ls(:,end))>hyperpar(i))) + 1/(options.firm-1)*sum((abs(ls(:,end))>hyperpar(i)).*(abs(ls(:,end))<=options.firm*hyperpar(i)));
            else
                beta_chap(:,i)= ls(:,i) ;
                df(i) = p ;
            end
        case 'scad'
            if (hyperpar(i)~=0)
                beta_chap(:,i)=lasso(:,i).*(abs(ls(:,i))<=2*hyperpar(i))+(options.scad*lasso(:,i)-ls(:,i))/(options.scad-2).*(abs(ls(:,i))>2*hyperpar(i)).*(abs(ls(:,i))<=options.scad*hyperpar(i))+ls(:,i).*(abs(ls(:,i))>options.scad*hyperpar(i));
                df(i) = sum((abs(ls(:,end))>hyperpar(i))) + 1/(options.scad-2)*sum((abs(ls(:,end))>2*hyperpar(i)).*(abs(ls(:,end))<=options.scad*hyperpar(i)));
            else
                beta_chap(:,i)= ls(:,i) ;
                df(i) = p ;
            end
        case {'adaptive','adalasso','garrote'}
            beta_chap(:,i)=lasso(:,i)+hyperpar(i)*sign(ls(:,i)).*(1-(hyperpar(i)./(abs(ls(:,end)))).^options.ada);
            df(i) = sum((abs(lasso(:,i))>hyperpar(i))) ;
        case 'enet'
            beta_chap(:,i)=lasso(:,i)*sqrt(1+options.enet);
            df(i) = sum((abs(lasso(:,i))>hyperpar(i)))*sqrt(1+options.enet) ;
        case 'adanet'
            beta_chap(:,i)=(lasso(:,i)+hyperpar(i)*sign(ls(:,i)).*(1-(hyperpar(i)./(abs(ls(:,end)))).^options.ada))*sqrt(1+options.enet);
            df(i) = sum((abs(lasso(:,i))>hyperpar(i)))*sqrt(1+options.enet) ;
    end
end
