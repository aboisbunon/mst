% Example of design

clear all
close all

addpath(genpath('..'))

%% MODEL DESIGN

% Size of the model
n = 750 ; % Number of observations
P = 250 ; % Number of variables

% Construction of an orthogonal design matrix X
% nom_X = ['mat_X' num2str(P) '_n' num2str(n) '.csv'] ;
% try
%     X = csvread(nom_X) ;   % if the matrix exits, load it
% catch ME
%     [Q,R] = qr(randn(n,P));
%     X = Q(:,1:P) ;
    X = randn(n,P);
%     csvwrite(nom_X,X) % if we want to save the matrix
% end

% Construction of parameter beta
k = 40 ;                     % Number of non-zero coefficients
optionsBeta.distr = 'unif' ; % Distribution of paramater beta ('norm' for a Gaussian, 
                             % 'unif' for a uniform on an interval)
optionsBeta.sigma = 2.5 ;      % Standard error or factor of the unit interval
optionsBeta.beta0 = 7.5 ;      % Constant to be added if we want mean(beta) non zero
beta = zeros(P,1) ;
if strcmp(optionsBeta.distr,'gauss')
    beta(randperm(P,k)) = (optionsBeta.sigma*randn(0,1,k ,1) + optionsBeta.beta0);
else
    beta(randperm(P,k)) = (optionsBeta.sigma*eval([optionsBeta.distr 'rnd(0,1,' num2str(k) ',1)']) + optionsBeta.beta0);
end

% Construction of error
options.type = 'gauss' ;   % Distribution of error ('unifSS' for a uniform  
                             % on a sphere with radius R, 'gauss' for a Gaussian)
options.sigma = 1 ;          % Radius of the sphere
epsilon = randSS(n,1,options) ;

% Construction of y
y = X*beta + epsilon ;

%% MODEL SELECTION

% Name of methods to use
explor = 'adalasso'; % for exploration (i.e. building the collection of models)
estim = 'ls'; % for estimation of the parameters
evalcrit = 'd0'; % for evaluation of the solutions
options.firm = 2; % some hyperparameters to be set

% The function for model selection
% If only final result is desired:
beta_best = ModSel(X,y,explor,estim,evalcrit,options) ;

% If more information is required:
[beta_best,beta_chap,crit,df] = ModSel(X,y,explor,estim,evalcrit,options) ;

% Visualization
[valmin,indmin] = min(crit);

figure(1)
plot((0:P),crit,'k')
xlabel('No of selected variables')
ylabel('loss (logarithmic scale)')
hold on
title(['Number of non zero coefficient k=' num2str(k)])
plot(indmin,valmin,'ok','MarkerFaceColor','black')
hold off
text(indmin,valmin+45,['k=' num2str(indmin)],'HorizontalAlignment','center')

%% COMPARISON OF METHODS OF ESTIMATION

explor1 = 'mcp'; % for exploration (i.e. building the collection of models)
estim1 = 'mcp'; % for estimation of the parameters
explor2 = 'lasso'; % for exploration (i.e. building the collection of models)
estim2 = 'lasso'; % for estimation of the parameters
evalcrit = 'd0'; % for evaluation of the solutions
options.firm = 2; % some hyperparameters to be set

[best_mcp,beta_mcp,crit_mcp,df_mcp] = ModSel(X,y,explor1,estim1,evalcrit,options) ;
[best_lasso,beta_lasso,crit_lasso,df_lasso] = ModSel(X,y,explor2,estim2,evalcrit,options) ;

% Visualization
[valmin_mcp,indmin_mcp] = min(crit_mcp);
[valmin_lasso,indmin_lasso] = min(crit_lasso);

figure(2)
plot((0:P),crit_mcp,'k')
hold on
plot((0:P),crit_lasso,'m')
xlabel('No of selected variables')
ylabel('loss (logarithmic scale)')
title(['Number of non zero coefficient k=' num2str(k)])
plot(indmin_mcp,valmin_mcp,'ok','MarkerFaceColor','black')
plot(indmin_lasso,valmin_lasso,'om','MarkerFaceColor','magenta')
hold off
text(indmin_mcp,valmin_mcp+45,['k=' num2str(indmin_mcp)],'HorizontalAlignment','center')
text(indmin_lasso,valmin_lasso+45,['k=' num2str(indmin_lasso)],'HorizontalAlignment','center')
legend('MCP','Lasso')

%% COMPARISON OF METHODS OF EVALUATION

explor = 'adalasso'; % for exploration (i.e. building the collection of models)
estim = 'ls'; % for estimation of the parameters
evalcrit1 = 'd0'; % for evaluation of the solutions
evalcrit2 = 'gcv'; % for evaluation of the solutions
options.firm = 2; % some hyperparameters to be set

[best_d0,beta_d0,crit_d0,df_d0] = ModSel(X,y,explor,estim,evalcrit1,options) ;
[best_gcv,beta_gcv,crit_gcv,df_gcv] = ModSel(X,y,explor,estim,evalcrit2,options) ;

% Visualization
[valmin_d0,indmin_d0] = min(crit_d0);
[valmin_gcv,indmin_gcv] = min(crit_gcv);

figure(3)
plot((0:P),crit_d0,'k')
hold on
plot((0:P),crit_gcv,'m')
xlabel('No of selected variables')
ylabel('loss (logarithmic scale)')
title(['Number of non zero coefficient k=' num2str(k)])
plot(indmin_d0,valmin_d0,'ok','MarkerFaceColor','black')
plot(indmin_gcv,valmin_gcv,'om','MarkerFaceColor','magenta')
hold off
text(indmin_d0,valmin_d0+45,['k=' num2str(indmin_d0)],'HorizontalAlignment','center')
text(indmin_gcv,valmin_gcv+45,['k=' num2str(indmin_gcv)],'HorizontalAlignment','center')
legend('L_0','GCV')