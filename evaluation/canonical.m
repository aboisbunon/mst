% Function transforming a linear model y = X*beta+error 
% in the canonical form.
%
% Input:
% - X = matrix of R^n*R^p
% - y = vector of R^n
%
% Output:
% - Z = vector of R^p
% - U = vector of R^{n-p}
%       => (Z;U) = G*y
% - A = matrix of R^p*R^p for the transformation from beta 
% to theta (theta = A*beta)
%
% A. Boisbunon, 09/2010
%

% function [Z,U,G1,G2,A] = canonical(y,X)
% 
%     [n,p] = size(X) ; % Tailles du modele
%     [Q,R] = qr(X)   ; % Decomposition QR de la matrice X
%     G1 = Q(:, 1:p)'  ;
%     G2 = Q(:, (p+1):n)';
%     A  = R(1:p, 1:p) ;
%     
%     Z = G1*y ;
%     U = G2*y ;
%

    
function [Z,U,A] = canonical(X,y)

    [n,p]	= size(X) ; % Size of the model
    [ZU,A]	= qr(sparse(X),y) ;
    
    A = full(A(1:p,1:p));
    Z = ZU(1:p,:);
    U = ZU((p+1):n,:) ;
    
end