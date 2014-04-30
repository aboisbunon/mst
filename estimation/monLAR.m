% Function for estimating the parameter beta of a linear model
% y = X*beta + error via the Lasso.
%
% Input:
% - X = design matrix (R^n*R^p),
% - y = study variable (R^n).
%
% Minimization problem of the Lasso:
% min_{beta} [||y-X*beta||^2 + lambda*|beta|],
% lambda >= 0 ;
%
% This function begins by setting beta_chap to 0, then finds
% what is the minimum value of lambda such that one coefficient 
% of beta becomes non zero, i.e. such that one variable of X
% enters the subset of selected variables. 
% This process is repeated until lambda = 0 (giving the ordinary
% least-squares solution).
%
% We also compute here the least-squares estimate reduced to 
% the selection performed by Lasso.
%
% Output:
% - beta_LAR = K cells (K = number of steps) with the following
%              parameters at step i:
%           o beta_LAR{i}.Beta    = Lasso estimate for beta
%           o beta_LAR{i}.indxsup = Lasso regularization path
% - beta_LS  = K cells (K = number of steps) with the following
%              parameters at step i:
%           o beta_LS{i}.Beta     = LS estimate for beta with subset
%           indxsup
%           o beta_LS{i}.indxsup  = Lasso regularization path
% - lambda   = hyperparamater (vector)
%
%
% A. Boisbunon, 02/2011
% Modified : 11/2011

function [beta_LAR, beta_LS, lambda] = monLAR(X,y, normalize)

if nargin<3
    normalize = 0 ;
end

[n, p]	= size(X) ;
if normalize
    X = X-repmat(mean(X),n,1) ;
    normX = sqrt(sum(X.^2,1));
    X = X./repmat(normX,n,1) ;
    y = y-mean(y);
end

% Initialization
indxsup{1}	= [] ;  % Selected variables
I0{1}	= 1:p ;             % Non selected variables
beta_LAR= zeros(p,p+1) ;  % Lasso estimate for beta
beta_LS = zeros(p,p+1) ;  % LS estimate for beta

r = X'*y ;      % Correlations with y
[lambda(1),i1] = max(abs(r)) ; % Search for the 1st var to be added
indxsup{2} = i1 ; I0{2} = I0{1} ; I0{2}(i1) = [] ; % Update
nI = 1 ;        % length of indxsup 
Xord = X(:,[indxsup{2} I0{2}]) ;
M = Xord'*Xord ;% Correlation matrix of X
S = 1/M(1,1) ;  % Inverse of M
v = Xord'*y/lambda(1) ;     % Subderivative of ||beta||_1

k = 1 ; % Current step
a_enlever = 0 ;

while(length(indxsup{k+1}) < p)&&(lambda(k)>=0)
    
    % Search for the next variable to be added
    tmp = S*sign(v(1:nI)) ;
    l1  = lambda(k)*(v(nI+1:end) -1)./(1-M(nI+1:end,1:nI)*tmp) + lambda(k) ;
    l1m = -lambda(k)*(v(nI+1:end) + 1)./(1+M(nI+1:end,1:nI)*tmp) + lambda(k) ;
    lmb1 = max(l1((l1<lambda(k))&(l1>=0))) ; lmb1m = max(l1m((l1m<lambda(k))&(l1m>=0))) ;
    
    % Update of lambda and beta
    switch ~isempty(lmb1)
        case 0
            switch ~isempty(lmb1m)
                case 0
                    lambda(k+1) = lambda(k);
                    i_next = [] ;
                case 1
                    lambda(k+1) = lmb1m ;
                    i_next = find(l1m==lambda(k+1));
                    i_next = i_next(1) ;
            end
        case 1
            switch ~isempty(lmb1m)
                case 0
                    lambda(k+1) = lmb1;
                    i_next = find(l1==lambda(k+1));
                    i_next = i_next(1) ;
                case 1
                    lambda(k+1) = max(lmb1,lmb1m) ;
                    if (lambda(k+1)==lmb1)
                        i_next = find(l1==lambda(k+1));
                        i_next = i_next(1) ;
                    else
                        i_next = find(l1m==lambda(k+1));
                        i_next = i_next(1) ;
                    end
            end
    end
    
    if (~a_enlever)
        beta_LAR(indxsup{k+1},k+1)	 = [beta_LAR(indxsup{k},k);0] + (lambda(k)-lambda(k+1))*S*sign(v(1:nI)) ;
    else
        beta_LAR(indxsup{k+1},k+1)	 = beta_LAR(indxsup{k}(setdiff(1:nI+1,i0))) + (lambda(k)-lambda(k+1))*S*sign(v(1:nI)) ;
    end
    
    if (k>1)&& isempty(i_next) % Step too long: one variable needs to be deleted
        
        % Search for the next variable to be deleted        
        l0  = lambda(k) + beta_LAR(indxsup{k+1},k+1)./(1+tmp) ;
        
        if ~isempty(l0((l0<lambda(k))&(l0>=0)))
            lambda(k+1) = max(l0((l0<lambda(k))&(l0>=0))) ;
            i0 = find(l0==lambda(k+1)) ; % Index of the variable to delete

            % Update of lambda and beta
            beta_LAR(indxsup{k},k+1)	 = beta_LAR(indxsup{k},k) + (lambda(k)-lambda(k+1))*S*sign(v(1:nI)) ;
            beta_LAR(indxsup{k}(i0),k+1) = 0 ;

            % Update of step k and subsets 
            k = k + 1 ;
            I0{k+1} = [I0{k} indxsup(i0)] ;
            indxsup{k+1} = indxsup{k} ; indxsup{k+1}(i0) = [] ;
            Xord = X(:,[indxsup{k+1} I0{k+1}]) ;
            nI = nI - 1 ;

            % Update of mu=X*beta_chap, residuals, subderivative and correlation matrix of X 
            if(isempty(indxsup{k}))
                mu = 0 ;
            else
%                mu = X(:,indxsup{k})*beta_LAR{k}.Beta ;
                mu = X*beta_LAR(:,k) ;
            end
            r = y - mu ;
            v = (Xord'*r)/lambda(k) ;
            M = Xord'*Xord ;
            S = reverseWoodBury(M(1:nI+1,1:nI+1),S) ;

            a_enlever = 1 ;
        else
            lambda(k+1) = 0 ;
        end
    else
        % Update of step k, subsets and inverse of M
        k = k + 1 ;
        
        fact = 1/(M(nI+i_next,nI+i_next)-M(nI+i_next,1:nI)*S*M(1:nI,nI+i_next));
        S = [S+fact*S*M(1:nI,nI+i_next)*M(nI+i_next,1:nI)*S -fact*S*M(1:nI,nI+i_next);
            -fact*M(nI+i_next,1:nI)*S fact];
        indxsup{k+1}	= [indxsup{k} I0{k}(i_next)] ;
        I0{k+1} = I0{k} ; I0{k+1}(i_next) = [] ;
        Xord = X(:,[indxsup{k+1} I0{k+1}]) ;
        nI	= nI + 1 ;

        % Update of mu=X*beta_chap, residuals, subderivative and correlation matrix of X 
        if(isempty(indxsup{k}))
            mu = 0 ;
        else
%                mu = X(:,indxsup{k})*beta_LAR{k}.Beta ;
            mu = X*beta_LAR(:,k) ;
        end
        r = y - mu ;
        v = (Xord'*r)/lambda(k) ;
        M = Xord'*Xord ;
        a_enlever = 0 ;
    
    end
    beta_LS(indxsup{k},k)	= X(:,indxsup{k})\y ;
    
end

beta_LS(indxsup{k+1},k+1) = X(:,indxsup{k+1})\y ;
beta_LAR(indxsup{k+1},k+1)= [beta_LAR(indxsup{k},k);0] + lambda(k)*S*sign(v(1:nI)) ;
lambda = [lambda';0] ;

end