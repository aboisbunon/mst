% Function solving Zhang's Minimax Concave Penalty (MCP) estimator in the 
% linear model: 
%       y = X*beta + error.
% MCP optimization problem is written as
%       min { ||Y-X*beta||^2 + pen(beta) },
% with pen(beta)= sum_j rho(|beta_j|) and
%       rho(|beta_j|) = (lambda*|beta_j| - beta_j^2/(2*gamma)
%                                           if  |beta_j|<= gamma*lambda
%                     = gamma*lambda/2      otherwise.
%
% INPUT
%   - X = design matrix of size (n,p)
%   - y = vector of size n
%   - gamma = hyperparameter tuning the bias (default = 2)
%   - normalize = boolean ; 1 if the user wants the data to be
%   normalized before applying MCP, 0 otherwise (default = 0).
%
% OUTPUT
%   - beta_MCP = matrix of size (p,k), k being the number of steps.
%   - lambda = vector of size k, the value of the hyperparameter at each
%   change in the state of beta.
%   - df = vector of size k, degrees of freedom computed for each step.
%   - ind_nextVar = vector of size p+1, containing the last step for which
%   beta_MCP has exactly p0 nonzero coefficients, 0 <= p0 <= p.
%
% A. Boisbunon, 01/2012
% Modified 09/2012
% 

function [beta_MCP, lambda, df, ind_nextVar] = monMCP(X, y, gamma, normalize) %,options

if nargin<3
    gamma = 2 ;
    normalize = 0 ;
end
if nargin<4
    normalize = 0 ;
end

[n, p]	= size(X) ;

% Normalization
if normalize
    X = X-repmat(mean(X),n,1) ;
    normX = std(X);
    X = X./repmat(normX,n,1) ;
    y = y-mean(y);
end

% Are there eigenvalues of X'*X very close to 1/gamma?
test = (eig(X'*X)-1/gamma);
if (abs(test)<10^-2)
    gamma = 2/(1-max(max(abs(X'*X)-diag(diag(abs(X'*X),0))))/n);
end

% Initialization
I0	= (1:p)' ;              % Index of non selected variables
I1 = [] ; I2 = [] ;         % Index of selected variables : I1 = penalized, I2 = non penalized
taille = 2*(p+1) ;          % Number of steps
beta_MCP	= zeros(p,taille) ;  % MCP estimate for beta
lambda = zeros(taille,1);   % Hyperparameter tuning the sparsity
df = zeros(taille,1);       % Degrees of freedom 

% FIRST STEP
[lambda(1),j] = max(abs(X'*y)); % j = first variable to add
move = 1 ;  % move = 1 means j has to go from I0 to I1
alpha = X'*y/lambda(1) ;    % One subgradient of the penalty at 0
k = 1 ;     % Current step
epsilon = 10^-6;    % precision
ind_nextVar = zeros(p+1,1) ; 

while (lambda(k)>0)
    [I2,I1,I0] = update_subsets(I2,I1,I0,j,move); % Update the subsets according to the change to perform
    I = [I2;I1]; % Total subset of non-zero coefficients in beta (ie of selected variables)
    n2 = length(I2); n1 = length(I1); n0=length(I0);
    
    % Update of s:
    % s_j = 0 if j belongs to I2, s_j = sign(beta_j) if j belongs to I1
    % s_j = sign(alpha_j) if j belongs to I0
    if move == 1
        s = [zeros(n2,1) ; sign(beta_MCP(I1(1:end-1),k));sign(alpha(I1(end)))] ; 
        ind_nextVar(n2+n1) = k ;
    else
        s = [zeros(n2,1) ; sign(beta_MCP(I1,k))] ;
    end
    
%     w = (X(:,I)'*X(:,I)-1/gamma*[zeros(n2,n2+n1);
%         zeros(n1,n2) eye(n1)])\s ;
    w = (X(:,I)'*X(:,I)-1/gamma*[zeros(n2,n2+n1);
        zeros(n1,n2) eye(n1)]+epsilon*eye(length(I)))\s ;
    w = w(:);
    z = X(:,I0)'*X(:,I)*w ; 
    
    % Compute degrees of freedom according to Zhang's formula
    df(k+1)=trace(inv(X(:,I)'*X(:,I)-1/gamma*[zeros(n2,n2+n1);
        zeros(n1,n2) eye(n1)])*(X(:,I)'*X(:,I)));
    
    % Update of lambda: which value for the next change?
    if ~isempty(I0) % Is there an index that should go from I0 to I1?
        lambda_possibles = lambda(k)*(alpha(I0)-1)./(1-z);
        lambda_possibles = [lambda_possibles;-lambda(k)*(alpha(I0)+1)./(1+z)];
    else
        lambda_possibles = [] ;
    end
    if ~isempty(I) % Is there an index that should go from I2 to I1 or from I1 to I2?
        lambda_possibles =[lambda_possibles;(beta_MCP(I,k)-gamma*lambda(k)*s)./(gamma*s+w)];
    end
    if ~isempty(I1)% Is there an index that should go from I1 to I0?
        lambda_possibles =[lambda_possibles;beta_MCP(I1,k)./w(length(I2)+1:end)];
    end
    lambda_possibles = lambda_possibles + lambda(k) ;
    % Which change occurs first?
    if ~isempty(lambda_possibles(lambda_possibles<lambda(k)-epsilon))
        lambda(k+1) = max(0,max(lambda_possibles(lambda_possibles<lambda(k)-epsilon)));
    else
        lambda(k+1) = 0 ;
    end
    tmp = find(lambda_possibles==lambda(k+1));
    if isempty(tmp)
        j = []; move = 0 ; lambda(k+1)= 0;
    else
        if tmp<=2*n0
            j = mod(tmp,n0) ; move = 1 ; 
            if j==0
                j= n0;
            end
        else
            if tmp <= 2*n0+n2
                j = tmp - 2*n0; move = -2;
            else
                if tmp <= 2*n0+n2+n1
                    j = tmp - (2*n0+n2); move = 2;
                else
                    j = tmp - (2*n0+n2+n1); move = -1;
                end
            end
        end
        if length(j)>1
            j = min(j(j~=0));
        end
    end
    
    % Update subgradient and beta
%     if lambda(k+1)~=0
        alpha(I0) = 1/lambda(k+1)*(lambda(k)*alpha(I0)+(lambda(k+1)-lambda(k))*z);
        beta_MCP(I,k+1) = beta_MCP(I,k)+(lambda(k)-lambda(k+1))*w;
%     end

    k = k+1;

    % If k is close to taille, we increase the sizes of matrices and
    % vectors to update
    if mod(k,2*p)==0
        beta_MCP = [beta_MCP zeros(p,taille)] ;
        lambda = [lambda;zeros(taille,1)] ;
        df = [df;zeros(taille,1)] ;
    end
end

% Final step
if sum(beta_MCP(:,k)==0)>0
    beta_MCP(:,k+1)= X\y ;
    df(k+1) = p;
    lambda(k+1) = 0 ;
    ind_nextVar(p) = k ;
    ind_nextVar(p+1) = k+1 ;
    k=k+1;
else
    ind_nextVar(p+1) = k ;
end
beta_MCP(:,k+1:end)=[];
lambda(k+1:end)=[];
df(k+1:end)=[];
ind_nextVar(ind_nextVar==0) = [];

end


function [I2_next,I1_next,I0_next] = update_subsets(I2_current,I1_current,I0_current,j,move)%p,

% if nargin == 1
%     I2_next = [] ; I1_next = [] ; I0_next = (1:p)';
% else

    I2_current = I2_current(:);
    I1_current = I1_current(:);
    I0_current = I0_current(:);

    switch move
        case 1 % From I0 to I1
            I2_next = I2_current ;
            I1_next = add_elt(I1_current,I0_current(j),'end') ;
            I0_next = move_elt(I0_current,I0_current(j),'out','value') ;
        case 2 % From I1 to I2
            I2_next = add_elt(I2_current,I1_current(j),'end') ;
            I1_next = move_elt(I1_current,I1_current(j),'out','value') ;
            I0_next = I0_current ;
        case -1 % From I1 to I0
            I2_next = I2_current ;
            I1_next = move_elt(I1_current,I1_current(j),'out','value') ;
            I0_next = add_elt(I0_current,I1_current(j),'start') ;
        case -2 % From I2 to I1
            I2_next = move_elt(I2_current,I2_current(j),'out','value') ;
            I1_next = add_elt(I1_current,I2_current(j),'start') ;
            I0_next = I0_current ;
        case 0 % No change in subsets
            I2_next = I2_current ;
            I1_next = I1_current ;
            I0_next = I0_current ;           
    end
% end
end


function [final_set] = add_elt(init_set, elt, pos)

nn = length(init_set);
switch pos
    case 'start'
        pos = 1 ;
    case 'end'
        pos = nn+1;
end
if pos>nn+1
    error('Element or position exceeds matrix dimensions')
end
init_set = init_set(:) ;
    
if isnumeric(pos)
    final_set = [init_set(1:pos-1);elt;init_set(pos:end)] ;
else
    if strcmp(pos,'start')
        final_set = [elt;init_set] ;
    else
        final_set = [init_set;elt] ;
    end
end

end


function [Pp] = move_elt(P, elt, pos, move_type, dim)

% move_type = 'index' --> permute elt'th row or column 
% move_type = 'value' --> permute the row or column with elements 
%                    equal to elt


nn = size(P);

if nargin==3
    if (length(elt)==1)&((elt-round(elt))==0)
        move_type = 'index' ;
    else
        move_type = 'value';
    end
end
if nargin<=4 
    if sum(nn==1)==0
        dim = 2 ;
    else
        dim = find(nn~=1);
    end
end

switch move_type 
    case 'index'
        indice = elt ;
    case 'value'
        n_elt = size(elt);
        tmp = P - repmat(elt,nn(1)-n_elt(1)+1,nn(2)-n_elt(2)+1) ;
        indice = find(sum(tmp,3-dim)==0) ;    
end

switch pos
    case 'start'
        pos = 1 ;
    case 'end'
        pos = nn(dim);
    case 'out'
        pos = 0 ;
end
if (pos>nn(dim))|(indice>nn(dim))
    error('Element or position exceeds matrix dimensions')
end
if (dim==1)
    P=P' ;
end

if pos==0
    Pp = P ;
    Pp(:,indice) = [] ;
else    
    if (indice>pos)
        Pp = P(:,[(1:pos-1)';indice;setdiff((pos:nn(dim))',indice)]) ;
    else
        Pp = P(:,[setdiff((1:pos)',indice);indice;(pos+1:nn(dim))']) ;
    end
end

if (dim==1)
    Pp=Pp' ;
end

end