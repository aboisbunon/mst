% Function generating some examples taken from the literature generating 
% the matrix X and the coefficient beta in the linear regression model:
%
%   y = X * beta + epsilon
%
% Input
% -----
%   - exemple	= string with values 'wells', 'tibshirani1', 'tibshirani2',
%   'tibshirani3', 'tibshirani4','breiman1','breiman2','breiman3','zou0'
%   and 'zou1' (see source code for references)
%   - options = structure with the following elements :
%       o options.orthog = 1 if X is an orthogonal matrix 
%                        = 0 otherwise.
%       o options.Xfixed = 1 if X is kept fixed afterward
%                        = 0 otherwise.
%       o options.rho  : max. correlation between variables in design
%       matrix X. Default value : 0.5.
%       o options.rhofixe = 1 if all pairs of variables have the same
%       correlation
%                         = 0 otherwise.
%       o options.snr  : signal-to-noise ratio desired (only for examples
%       from Breiman). 
%       o options.rc  : radius of clusters of coefficients (only for examples
%       from Breiman). 
% 
% Output
% ------
%   - X     = either the true design matrix of size n*p if
%   Xfixed=1, or the covariance matrix of size p*p if Xfixed=0.
%   - beta  = true coefficient of regression (vector of length p).
%
% A. Boisbunon, 08/2012


function [X,beta] = ex_literature(exemple,options)

if ~isfield(options,'rho')
    options.rho = 0.5;            % Maximum correlation between variables in design matrix X
end
if ~isfield(options,'Xfixed')
    options.Xfixed = 1;            % Design matrix X: fixed=1, random=0
end
if ~isfield(options,'snr')
    if options.Xfixed 
        options.snr = .85 ;
    else
        options.snr = .75 ;
    end
end
if ~isfield(options,'rhofixe')
    options.rhofixe = 1;            % Pairwise correlation rho of X: fixed for all i,j=1, random=0
end
if ~isfield(options,'orthog')
    options.orthog = 1;
end

n = options.n ;

% Construct beta
switch exemple
   % From Fourdrinier & Wells (1994), Comparaisons de procedures de
   % selection d'un modele de regression: une approche decisionnelle
    case {'fourdrinier','wells','fw'}
        if ~isfield(options,'p')
            options.p = 5;            % Size of zeros at the end of beta
        end
        beta = [2;0;0;4;0;zeros(options.p-5,1)] ;
   % Example 1 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case 'tibshirani1'
        beta = [3;1.5;0;0;2;0;0;0] ;
   % Example 2 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case 'tibshirani2'
        beta = [.85;.85;.85;.85;.85;.85;.85;.85] ;
   % Example 3 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case 'tibshirani3'
        beta = [5;0;0;0;0;0;0;0] ;
   % Example 4 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case 'tibshirani4'
        beta = [zeros(10,1);repmat(2,10,1);zeros(10,1);repmat(2,10,1)] ;
   % Example in the X-controlled case from Breiman 
    case 'breiman1'
        p = 20 ;
        beta = zeros(p,1) ;
        for i=-options.rc:1:options.rc
            beta([i+5;i+15]) = repmat((options.rc-abs(i))^2,2,1) ;
        end
        options.rho = 2*rand(1)-1;
   % Example in the X-random case from Breiman
    case 'breiman2'
        p = 40 ;
        beta = zeros(p,1) ;
        for i=-options.rc:1:options.rc
            beta([i+10;i+20;i+30]) = repmat((options.rc-abs(i))^2,3,1) ;
        end
        options.rho = 2*rand(1)-1;
    case 'breiman3'
        p = 40 ;
        beta = zeros(p,1) ;
        beta(randsample(p,20)) = randn(1)*(rand(20,1)+1) ;
        options.rho = 2*rand(1)-1;
   % Example 0 from Zou (2006), the adaptive lasso and its oracle properties
    case 'zou0'
        beta = [5.6;5.6;5.6;0] ;
   % Example 1 from Zou (2006), the adaptive lasso and its oracle properties
    case 'zou1'
        p = round(4*sqrt(n))-5 ;
        q = round(p/9) ;
        beta = [3*ones(q,1);3*ones(q,1);3*ones(q,1);zeros(p-3*q,1)] ;
end

p = length(beta) ;

% Construct design matrix X
switch exemple
   % From Fourdrinier & Wells (1994), Comparaisons de procedures de
   % selection d'un modele de regression: une approche decisionnelle
    case 'martin'
        switch options.orthog
            case 0
                if options.rhofixe
                    Sigma = (1-options.rho)*eye(p)+options.rho*ones(p);
                else
                    test = 1 ; 
                    while test
                        Sigma = options.rho*rand(p);
                        Sigma = tril(Sigma)+ tril(Sigma)' ;
                        Sigma = Sigma + diag(1-diag(Sigma));
                        [T,test] = cholcov(Sigma);
                    end
                end
                X = mvnrnd(zeros(n,p),Sigma) ; % Generate explanatory variables (orthogonal design)
            case 1
                [Q,R1] = qr(randn(n,p)); % For an orthogonal design matrix X
                X = Q(:,1:p) ;
                clear R1
        end
   % Examples 1 to 3 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case {'tibshirani1','tibshirani2','tibshirani3','breiman1','breiman2','breiman3'}
        Sigma = eye(p);
        for i=2:p
            Sigma= Sigma + options.rho^(i-1)*diag(ones(p-i+1,1),i-1)+ options.rho^(i-1)*diag(ones(p-i+1,1),-i+1);
        end
        X = mvnrnd(zeros(n,p),Sigma);
        if strcmp(exemple,'breiman1')|strcmp(exemple,'breiman2')
            alpha = sqrt((n*options.snr/(1-options.snr)-1)/norm((eye(n)-ones(n)/n)*X*beta)^2);
            beta = beta*alpha; % Renormalization to obtain R^2=options.snr
        end
   % Example 4 from Tibshirani (1996), Regression shrinkage and selection via the
   % lasso
    case 'tibshirani4'
        Sigma = (1-options.rho)*eye(p)+options.rho*ones(p) ;
        if (options.rho==.5)
            X = randn(n,p) + repmat(randn(n,1),1,p) ;
        else
            X = mvnrnd(zeros(n,p),Sigma);
        end
   % Example 0 and 1 from Zou (2006), the adaptive lasso and its oracle properties
    case {'zou0','zou1'}
        rho1 = .39 ; rho2 = .23 ;
        Sigma = [(1-rho1)*eye(p-1)+rho1*ones(p-1) rho2*ones(p-1,1);rho2*ones(1,p-1) 1];
        X = mvnrnd(zeros(n,p),Sigma);
end

if ~options.Xfixed
    X = Sigma ;
end

end