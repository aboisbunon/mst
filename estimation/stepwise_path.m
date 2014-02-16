% Function computing stepwise methods (forward selection and backward
% elimination) for the linear regression model 
%   
%       y = X*beta + epsilon
%
% where y is the response vector, X is the design matrix, beta is the
% unknown regression coefficient we wish to estimate, and epsilon is the
% noise vector.
%
% Input
% -----
%   - X       = matrix of size n*p
%   - y       = vector of size n*1
%   - type  = string with one of the following values : 
%       o 'forward', 'f', 'for' or 'fd' for forward selection,
%       o 'backward', 'b', 'back', 'bd' for backward elimination
% 
% Output
% ------
%   - beta_chap = matrix of size p*(p+1) of estimators of beta for different
%               subsets.
%
% A. Boisbunon, 08/2012

function [beta_chap] = stepwise_path(X,y,type)

p=size(X,2);
beta_chap = zeros(p,p+1);

switch type
    case {'forward','f','fd','for'}
        ind = zeros(p,1) ;          % Order of selection of the variables
        [mm,ind(1)] = max(abs(X'*y)) ;
        beta_chap(ind(1),2) = X(:,ind(1))\y;
        reste = setdiff((1:p)',ind(1)) ;            % Non-selected variables
        for jj=2:p-1 % At each step
            mse_jj = sum((y-X*beta_chap(:,jj)).^2) ;

            % First: estimation of beta_LS with a new variable included
            beta_tmp = zeros(p,length(reste)) ;
            for j=1:length(reste) % We test each possible new variable
                beta_tmp([ind(1:jj-1);reste(j)],j) = X(:,[ind(1:jj-1);reste(j)])\y ;
            end
            % Then: we estimate the difference in mse for each new variable
            mse_tmp = sum((repmat(y,1,size(beta_tmp,2))-X*beta_tmp).^2) ;
            % Finally, we select the one with max difference in mse
            [mm,indtmp] = max(mse_jj - mse_tmp) ;
            ind(jj) = reste(indtmp);
            reste(indtmp) = [] ;

            beta_chap(:,jj+1) = beta_tmp(:,indtmp) ;

            clear beta_tmp mse_tmp indtmp
        end
        beta_chap(:,end) = X\y ;
    case {'backward','b','bd','back'}
        ind = (1:p)' ;          % Order of selection of the variables
        reste = zeros(p,1) ;            % Non-selected variables
        beta_chap(:,1)=X\y ;
        for jj=1:p-1 % At each step
            mse_jj = sum((y-X*beta_chap(:,jj)).^2) ;
           % First: estimation of beta_LS with a new variable included
            beta_tmp = zeros(p,length(ind)) ;
            for j=1:length(ind) % We test which variable to remove
                ind_tmp = ind(setdiff(1:length(ind),j));
                beta_tmp(ind_tmp,j) = X(:,ind_tmp)\y ;
            end

            % Then: we estimate the difference in mse for each variable
            mse_tmp = sum((repmat(y,1,size(beta_tmp,2))-X*beta_tmp).^2) ;
            % Finally, we select the one with min difference in mse
            [mm,indtmp] = min(mse_tmp-mse_jj) ;

            % Finally, we remove the one with lower estimator of loss
            reste(jj) = ind(indtmp) ;
            ind(indtmp) = [] ;
            beta_chap(:,jj+1) = beta_tmp(:,indtmp) ;

        end

%     case {'stagewise','s','stage','sw'}
%         ind = zeros(p,1) ;          % Order of selection of the variables
%         [mm,ind(1)] = max(abs(X'*y)) ;
%         beta_chap(ind(1),2) = X(:,ind(1))\y;
%         reste = setdiff((1:p)',ind(1)) ;            % Non-selected variables
%         for jj=2:p-1 % At each step
%             mse_jj = sum((y-X*beta_chap(:,jj)).^2) ;
% 
%             % First: estimation of beta_LS with a new variable included
%             beta_tmp = zeros(p,length(reste)) ;
%             for j=1:length(reste) % We test each possible new variable
%                 beta_tmp([ind(1:jj-1);reste(j)],j) = X(:,[ind(1:jj-1);reste(j)])\y ;
%             end
%             % Then: we estimate the difference in mse for each new variable
%             mse_tmp = sum((repmat(y,1,size(beta_tmp,2))-X*beta_tmp).^2) ;
%             % Finally, we select the one with max difference in mse
%             [mm,indtmp] = max(mse_jj - mse_tmp) ;
%             ind(jj) = reste(indtmp);
%             reste(indtmp) = [] ;
% 
%             beta_chap(:,jj+1) = beta_tmp(:,indtmp) ;
% 
%             clear beta_tmp mse_tmp indtmp
%         end
%         beta_chap(:,end) = X\y ;
end