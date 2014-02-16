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
%   - X       = matrix of size n*p
%   - y       = vector of size n*1
%   - beta_chap = matrix of size p*taille of estimators of beta for different
%               subsets, where taille is equal the number of subsets tested.
%   - df = vector of size taille  with the degrees of freedom for each solution.
%   - ls = least-squares solution.
%   - type = string for the name of criterion for model evaluation.
%   - var_estim = string for the type of estimator of the variance (Maximum
%   Likelihood or unbiased)
%   - full  = 1 if the variance is computed for the full model
%           = 0 if the variance is for each submodel.
%   - explor,estim and options = same as for ExplorEstim.m
% 
% Output
% ------
%   - crit = values for the relative performance of models.
%
% A. Boisbunon, 02/2013



function [crit] = EvalModel(X,y,beta_chap,df,ls,type,explor,estim,options,var_estim,full)


if (nargin<11)
    switch estim
        case {'enet','adanet'}
            full = 1 ;
        otherwise
            full = 0 ;
    end
end

if (nargin<10)
    switch estim
        case {'enet','adanet'}
            full = 1 ;
            var_estim = 'unb' ;
        otherwise
            full = 0 ;
            var_estim = 'unb' ;
    end
end

n = length(y) ;
df = df(:) ;
selec = sum((beta_chap~=0),1); selec = selec(:) ;

% Sum of squared error
residual = repmat(y,1,size(beta_chap,2))-X*beta_chap ;
logV = sum(residual.^2,1) ; logV = logV(:);

% ESTIMATORS OF VARIANCE
% ----------------------

SI = sum((repmat(y,1,size(ls,2))-X*ls).^2) ; SI = SI(:);
switch full
    case 1 % For computing the Least-squares on the full model
        ind = length(SI);
    case 0 % For  computing the Least-squares on the model restricted to the selection
        ind = 1:length(SI);
end
switch var_estim
    case {'ml','ML','maxlik'} % Maximum Likelihood variance estimator = ||Y-X*beta_OLS||^2/n
        sig = SI(ind)/n;
    case {'unb','sb','unbiased'} % Unbiased variance estimator = ||Y-X*beta_OLS||^2/(n-k)
        sig = SI(ind)./(n-selec(ind));
    case {'mod','model'} % Unbiased variance estimator = ||Y-X*beta_OLS||^2/(n-k)
        sig = logV./(n-df);
end
sig = sig(:);

switch type

    %       ---------------
    %       LOSS ESTIMATORS
    %       ---------------
    case {'d0','L0','unbiased','ul', 'UL','ule','ULE','delta0'} % Unbiased loss estimators
        crit = logV + (2*df - n).*sig  ;  
%        keyboard
    case {'d0inv','L0inv','deltaInv0'} % Unbiased estimators of the invariant loss
        crit = (n-selec(ind)-2)*logV/SI(ind) + 2*df - n - 4*diag(residual'*X*(beta_chap-ls))./SI(ind) ;
    	crit(selec==0) = Inf ;
    case {'dAst','Last'} % Corrected loss estimator by Fourdrnier and Wells (1994)
        % crit = (df./(n-df+2)- 2*(df-4)./((n-df+4).*(n-df+6)).*logV./sum((X*beta_chap).^2,1)').*logV  ;
        crit = selec(:)./(n-selec(:)+2).*logV- 2*(df-4)./((n-selec(:)+4).*(n-selec(:)+6))./sum((X*ls).^2,1)'.*SI.^2  ;
    case {'CE1', 'ce1','dGamma1'} % Corrected loss estimators by Boisbunon et al
        a1 = 2*logV.^2./((n-df).*(n-df+2))+4*diag((X*beta_chap-repmat(y,1,size(beta_chap,2)))'*X*ls) ;
        gamma1 = a1./sum((X*ls).^2)' ;      
        gamma1(selec==0) = 0 ;
        crit = logV + (2*df - n).*sig - gamma1 ; 
    case {'CE2', 'ce2','dGamma2'} % Corrected loss estimators by Boisbunon et al
        Z = zeros(p,taille) ;
        [Q,Rqr] = qr(X); clear 'Rqr'
        for j=1:taille
            Z(beta_chap(:,j)==0,j) = Q(:,beta_chap(:,j)==0)'*y; % To compute the correction function gamma
        end
        gamma2 = 1./( selec.*(selec+1).*max(Z.^2)'+sum(Z.^2)' ) ;      
        a2 = -S/(n-p)*(S/(n-p+2)*(-2*p+4*selec.*(selec+1))+4./gamma2);
        gamma2 = a2.*gamma2 ; gamma2(selec==0)=0;
        crit = logV + (2*df - n).*sig - gamma2 ;        
        
    % ------------------------------
    % OTHER CRITERIA FROM LITERATURE
    % ------------------------------

    %       CROSS VALIDATION
    %       ----------------

    case {'loocv','LOO','LOOCV','loo'} % Leave-on-out cross validation (LOOCV)
        for i=1:n
            beta_train = ExplorEstim(X(setdiff(1:n,i),:), y(setdiff(1:n,i),r), explor, estim, options) ;
            y_test = X(i,:)*beta_train ;
            crit = crit + (y_test'-y(i,r)).^2/n;
        end

    case {'cv5','CV5'} % 5-fold cross validation (CV5)
        indtrain = randperm(n) ;
        for i=1:5
            if i~=5
                test = indtrain((1:round(n/5))+(i-1)*round(n/5)) ;
            else
                test = indtrain((i-1)*round(n/5):end) ;
            end
            beta_train = ExplorEstim(X(setdiff(1:n,test),:), y(setdiff(1:n,test),r), explor, estim, options) ;
            y_test = X(test,:)*beta_train ;
            crit = crit + sum((y_test-repmat(y(test),1,size(y_test,2))).^2)'/5;
        end

    case {'cv10','CV10'} % 10-fold cross validation (CV10)
        indtrain = randperm(n) ;
        for i=1:10
            if i~=10
                test = indtrain((1:round(n/10))+(i-1)*round(n/10)) ;
            else
                test = indtrain((i-1)*round(n/10):end) ;
            end
            beta_train = ExplorEstim(X(setdiff(1:n,test),:), y(setdiff(1:n,test),r), explor, estim, options) ;
            y_test = X(test,:)*beta_train ;
            crit = crit + sum((y_test-repmat(y(test),1,size(y_test,2))).^2)'/10;
        end

    case {'gcv','GCV'} % Golub et al: Generalized Cross Validation
        crit = n*logV./(n-df).^2 ;
    case {'srm','SRM'} % Vapnik (1990): Structural Risk Minimization
    	crit = logV./max(1-sqrt(df/n.*(log(n./df)+1)-log(sqrt(n))/n),0) ;
    case {'aic','AIC'} % Akaike (1973): Akaike Information Criterion
        if full
            crit = logV./sig + 2*df  ;
        else
            crit = log(sig) + 2*df ; 
        end
    case {'aic3','AIC3'} % Bozdogan: AIC3
        if full
            crit = logV./sig + 3*df  ;
        else
            crit = log(sig) + 3*df ; 
        end
    case {'aicc','AICc'} % Sugiura (1978): corrected AIC
        if full
            crit = logV./sig + 2*df.*(df+1)./(n-df-1)  ;
        else
            crit = log(sig) + 2*df.*(df+1)./(n-df-1) ; 
        end
    case {'bic','BIC'} % Schwartz (1976): Bayes Information Criterion
        if full
            crit = logV./sig + log(n)*df  ;
        else
            crit = log(sig) + log(n)*df ; 
        end
    case {'cp','Cp'} % Mallows (1973): Cp
    	crit = logV./sig + 2*df - n ; 

    case {'slope','sh','SH'} % Birge and Massart: Slope Heuristics 
    	crit = slope_heuristic(X, y, beta_chap, df, options);
end
if (sum(crit == 0)~= 0)
    crit(crit == 0) = NaN ;
end


end
