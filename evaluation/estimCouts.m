% Function computing the unbiased and corrective estimators 
% of quadratic loss ||phi-theta||^2, where phi is the least-
% squares estimate on the subset selected by Lasso.
%
% Input: 
% - X = design matrix of R^n*R^P;
% - y = study variable (vector of R_n); 
% - beta_chap = estimate of regression coefficients  
%               (vector of R_p or matrix of R_p*R_{length(c)} 
%               if c = vector);
% - estim = name of estimator for parameter beta
%               ('ls' = least-squares ;
%                'lasso'/'lar' = estimate with the LAR algorithm
%                'firm' = Firm-threshold shrinkage
%                'scad' = SCAD penalty 
% - c = constant or vector corresponding to the Lasso hyperparameter;
% - a = constant of the corrective function gamma.
%
% This function first transforms the data under the canonical
% form before estimating loss:
% - phi = canonical form of beta_chap
% - z = vector of R_p ; 
% - u = vector of R_(n-p) ; G*y = (z;u)
%
% Output: 
% - deltaSB = unbiased loss estimator;
% - deltaGamma2 = corrective loss estimator; 
% - A = matrix transforming beta_chap into phi ;
% - p = number of selected variables for each value of c.
%
%
% A. Boisbunon, 03/2011
%


function [deltaSB, deltaGamma2, A, S] = estimCouts(X, y, beta_chap, c, df)
% function [deltaSB, deltaGamma2, A, p] = estimCouts(X, y, beta_chap, c)

    [n,P] = size(X) ; 
    df = df(:);
    c = c(:) ;

    % Canonical form of y and beta_chap
    [z,u,A] = canonical(X,y) ;
    phi = zeros(P, length(c)) ;
    for i = 1:1:length(c)
        phi(:,i) = A*beta_chap(:,i) ;
    end
    
    norm2_U = (norm(u,2))^2 ; % scalar
    
    % Computation of the corrective function
    Z = repmat(z,1,length(c)) ;
    C = repmat(c(:)',length(z),1) ;
    
    aZC = abs(Z) - C ;   % |z_j|-c(i), for all i=1,length(c)
    
    p   = sum(aZC>0) ;               % card{|z(j)|> c(i)}
    maxZ = max((aZC<=0).*abs(Z)) ;  % max_|z|{|z(j)|<= c(i)}
    nZ0 = sum((aZC<=0).*Z.^2) ;     % sum_j |z(j)| such that {|z(j)|<= c(i)}
   
 %   a = ((0:1:P)-3*P/5).*((0:1:P)-P)*norm2_U/(6*(n-P)^2*(n-P+4)*(n-P+6)) ;
 %   gamma2 = a.*(p~=P)./( (p~=P).*p.*maxZ.*maxZ+nZ0+(p==P) ) ;

    % Computation of function g such that phi(Z) = Z + g(Z)
    G = phi - Z;

    % Computation of loss estimators

    % Unbiased loss estimators
    deltaSB = (2*df-P)/length(u)*S + sum(G.*G)' ;

    % Corrected loss estimators
    a2 = 2/((n-P+4)*(n-P+6))*(P-2-2*(k+1).*(k/P)) ;
    gamma2 = a2.*(p~=P)./( (p~=P).*p.*maxZ.*maxZ+nZ0+(p==P) ) ;
    deltaGamma2 = deltaSB - gamma2*norm2_U^2 ;
    
    if (size(p,1)==1)
        p = p(:) ;
        deltaSB = deltaSB(:) ;
        deltaGamma2 = deltaGamma2(:);
    end
        
end
