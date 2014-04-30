% Function computing both exploration and estimation steps for a model
% selection procedure applied to the linear model
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
%   - beta_chap = matrix of size p*taille of estimators of beta for different
%               subsets, where taille is equal to 2^p for all subsets and to p+1
%               otherwise.
%   - ls        = Least-squares estimator for computing the variance.
%   - df        = vector of size taille*1 of the estimated degrees of freedom
%               (useful for evaluating the solutions).
%
% A. Boisbunon, 11/2011
% Modified : 08/2012

function [beta_chap, LS, df] = ExplorEstim(X, y, explor, estim, options)% = , subsets,hyperpar

if nargin<5
    options.orthog = 0 ;
end

if ~isfield(options,'orthog')
    options.orthog = 0;
end;
if ~isfield(options,'firm')
    options.firm = 2;
end;
if ~isfield(options,'enet')
    options.enet = 0.2;
end
if ~isfield(options,'ada')
    if strcmp(explor,'garrote')
        options.ada = 1;
    else
        options.ada = 2;
    end
end
if ~isfield(options,'nb_var')
    options.nb_var = 3;
end

[n,p] = size(X) ; % n = no of observations, p = no of variables in X

% Initialization of 'taille'
switch explor
    case'subset'
        if p>12 % If p has more than 12 variables, all subset is too long to compute
             error('myApp:argChk', 'Too many variables for exhaustive exploration')
        else
            taille = 2^p ;  % Number of possible submodels
        end
    case {'lasso','scad','adaptive','adalasso','garrote','enet','adanet','mcp','firm','backward','forward','backtype','fortype'}
        taille = p + 1 ;
    case 'wide_path'
        taille = (options.nb_var+1)*(p-1)+2;
end


%       ------------------------------
%       STEP 1: EXPLORATION OF SUBSETS
%       ------------------------------

switch explor
    case 'subset' % All subsets / exhaustive exploration 
        if p<=12
            subsets = [0];  
            for i=1:1:p
                subsets = [subsets ; num2cell(combntns(1:p,i),2)];
            end
            LS = zeros(p,taille) ;
            for j=2:taille
                LS(subsets{j},j) = X(:,subsets{j})\y ;
            end
        end
    case {'backward','forward'}
        LS = stepwise_path(X,y,explor) ;
%         ind_sel = sbs(X,y,0.05,'lin');
%         beta_chap = zeros(p,taille) ;
%         for j=2:length(ind_sel)+1
%             beta_chap(ind_sel(1:j-1),j) = X(:,ind_sel(1:j-1))\y ;
%         end
        df = sum((LS~=0),1)' ; df=df(:);
    case 'backtype'
        LS = proc_sel(X,y,'backward') ;
        df = sum((LS~=0),1)' ; df=df(:);
    case 'fortype'
        LS = proc_sel(X,y,'forward') ;
        df = sum((LS~=0),1)' ; df=df(:);
    case 'wide_path'
        beta_chap = zeros(p,taille) ;
        chem_elargi = replique_chemin(p,options.nb_var);
        [beta_path] = monMCP_moche(X,y,2,0) ;
        reg_path=[];
        for j=2:size(beta_path,2)
            reg_path = [reg_path;setdiff(find(beta_path(:,j)~=0),reg_path)];
        end
        for jj=1:p
            if jj~=p
                for j=1:options.nb_var+1
                    if j<=options.nb_var
                        ind_tmp = reg_path(chem_elargi{jj}(j,:));
                    else
                        ind_tmp = reg_path(1:jj);
                    end
                    beta_chap(ind_tmp,(jj-1)*(options.nb_var+1)+j+1) = X(:,ind_tmp)\y ;
                end
            else
                beta_chap(:,end) = X\y ;
            end
        end
        df = sort([0;repmat((1:p-1)',options.nb_var+1,1);p]);
    case {'firm' ,'lasso','mcp','adalasso','adaptive','garrote','enet','adanet','scad'}
        options.estim = estim ;
        if options.orthog == 1
            [beta_chap, LS, df] = Shrinkage(X,y,options) ;
        else
            switch explor
                case {'lasso','LASSO'}   
                    [beta_chap,LS] = monLAR(X,y,0) ;
                    df = (0:p); df = df(:) ;
                case {'mcp','firm'}
                    [beta_chap, mu, df,ind_next] = monMCP(X,y,options.firm,0) ;
                    %ind_next = [1;ind_next];
                    beta_chap = beta_chap(:,ind_next);
                    df = df(ind_next);
                    LS = zeros(size(beta_chap));
                    for etape = 1:size(beta_chap,2)
                        LS(beta_chap(:,etape)~=0,etape) = pinv(X(:,beta_chap(:,etape)~=0))*y;
                    end
                case {'adalasso','adaptive','garrote'}
                    betaLS = pinv(X)*y;
                    Xtilde = X.*repmat((betaLS(:).^options.ada)',n,1);
                    [beta_chap,LS,lambda] = monLAR(Xtilde,y) ;
                    beta_chap = beta_chap.*repmat((betaLS(:).^options.ada),1,p+1);
                    LS = LS.*repmat((betaLS(:).^options.ada),1,p+1);
                    df = [] ;
                case {'enet'}
                    Xtilde = [X;sqrt(options.enet)*eye(p)]/sqrt(1+options.enet);
                    ytilde = [y;zeros(p,1)];
                    [beta_chap,ridge] = monLAR(Xtilde,ytilde) ;
                    beta_chap = beta_chap*sqrt(1+options.enet);
                    ridge = ridge/sqrt(1+options.enet);
                    LS = zeros(size(beta_chap)); df = zeros(size(beta_chap,2),1) ;
                    for etape = 1:size(beta_chap,2)
                        LS(beta_chap(:,etape)~=0,etape) = pinv(X(:,beta_chap(:,etape)~=0))*y;
                        df(etape) = trace((X(:,beta_chap(:,etape)~=0)'*X(:,beta_chap(:,etape)~=0)+options.enet)\(X(:,beta_chap(:,etape)~=0)'*X(:,beta_chap(:,etape)~=0)));
                    end
                    df = df(:) ;
                case {'adanet'}
                    betaLS = pinv(X)*y;
                    Xtilde = [X.*repmat((betaLS(:).^options.ada)',n,1);sqrt(options.enet)*eye(p)]/sqrt(1+options.enet);
                    ytilde = [y;zeros(p,1)];
                    [beta_chap,ridge] = monLAR(Xtilde,ytilde) ;
                    beta_chap = beta_chap.*repmat((betaLS(:).^options.ada),1,p+1)*sqrt(1+options.enet);
                    ridge = ridge.*repmat((betaLS(:).^options.ada),1,p+1)*sqrt(1+options.enet);
                    LS = zeros(p,p+1);
                    for etape = 1:p+1
                        LS(beta_chap(:,etape)~=0,etape) = pinv(X(:,beta_chap(:,etape)~=0))*y;
                    end
                    df = [] ;
                case 'scad'
                    error('estim:type','Estimator not programmed yet')
            end
            
        end
        
end


%       -------------------------------------------------
%       STEP 2: ESTIMATION OF BETA ON SUBSETS FROM STEP 1 
%       -------------------------------------------------

% (only if 'estim' different from 'explor', otherwise beta_chap has already
% been estimated in Step 1)

if ~strcmp(explor,estim)
    df = sum((LS~=0),1) ; 
    beta_chap = zeros(p,taille) ;
    switch estim
        case {'ls','LS'}
            beta_chap = LS ;
            df = df(:) ;
        case {'js','JS'}
            beta_chap(:,sum((beta_chap==0),2)~=p) = LS(:,sum((beta_chap==0),2)~=p)-repmat((df(sum((beta_chap==0),2)~=p)-2)./sum(LS(:,sum((beta_chap==0),2)~=p).^2),p,1).*LS(:,sum((beta_chap==0),2)~=p) ;
            df(sum((beta_chap==0),2)~=p) = df(sum((beta_chap==0),2)~=p)-(df(sum((beta_chap==0),2)~=p)-2).^2./sum(X*LS(:,sum((beta_chap==0),2)~=p).^2); df = df(:) ;
        case {'gjs','GJS'}
            beta_chap(:,sum((beta_chap==0),2)~=p) = LS(:,sum((beta_chap==0),2)~=p)-repmat((df(sum((beta_chap==0),2)~=p)-2).*sum((repmat(y,1,size(LS(:,sum((beta_chap==0),2)~=p),2))-X*LS(:,sum((beta_chap==0),2)~=p)).^2)./((n-df(sum((beta_chap==0),2)~=p)-2).*sum(X*LS(:,sum((beta_chap==0),2)~=p).^2)),p,1).*LS(:,sum((beta_chap==0),2)~=p) ;
            df(sum((beta_chap==0),2)~=p) = df(sum((beta_chap==0),2)~=p)-(df(sum((beta_chap==0),2)~=p)-2).^2.*sum((repmat(y,1,size(LS(:,sum((beta_chap==0),2)~=p),2))-X*LS(:,sum((beta_chap==0),2)~=p)).^2)./((n-df(sum((beta_chap==0),2)~=p)-2).*sum(X*LS(:,sum((beta_chap==0),2)~=p).^2)); df = df(:) ;
        case {'ridge', 'Ridge','rr','RR','RIDGE'}
            for etape = 2:p+1
                H = inv(X(:,LS(:,etape)~=0)'*X(:,LS(:,etape)~=0)+options.enet)*X(:,LS(:,etape)~=0)';
                beta_chap(LS(:,etape)~=0,etape) = H*y;
                df(etape) = trace(X(:,LS(:,etape)~=0)*H); 
            end
            df = df(:) ;
    end
end
end
