%% This example shows how to generate spherically symmetric random vectors.

clear all
close all

addpath(genpath('..'))

%% Parameters
N = 2 ; % Length of each random vector
M = 10000 ; % Number of random vectors to generate

%% Data generation

% Example 1 : multivariate spherical Gaussian 
options.type = 'gauss' ; % Type of distribution
options.scale = 1 ; % Scale parameter
Xgauss = randSS(N,M,options) ;

% Example 2 : multivariate spherical Student 
options.type = 't_mg' ; % Type of distribution
options.df = 5 ; % Number of degrees of freedom
Xstudent = randSS(N,M,options) ;

% Example 3 : multivariate spherical Kotz 
options.type = 'kotz' ; % Type of distribution
options.N_kotz = 2 ; options.r = .5 ; options.s = 1 ; % Parameters of the Kotz distribution
Xkotz = randSS(N,M,options) ;

% Example 4 : multivariate spherical Laplace 
options.type = 'laplace' ; % Type of distribution
options.scale = 1 ; % Scale parameter
Xlaplace = randSS(N,M,options) ;

% Example 5 : multivariate spherical (continuous) Bessel 
options.type = 'bessel' ; % Type of distribution
options.q = 1 ; options.r = 1 ; options.scale = 1 ; % Parameters of the Bessel distribution
Xbessel = randSS(N,M,options) ;

% Example 6 : multivariate spherical exponential power 
options.type = 'exp_power' ; % Type of distribution
options.power = 3 ; % Power parameter
Xpower = randSS(N,M,options) ;

%% Visualization

soft = whichsoft();
if strcmp(soft,'Octave')
    [desc, flag] = pkg ('describe', 'statistics');
    if ~strcmp(flag,'Loaded')
        pkg install -forge -nodeps statistics
    end
    pkg load statistics
end


figure(1)

% Gaussian 
subplot(3,2,1)
hist3(Xgauss',[30 30])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Gaussian')

% Student 
subplot(3,2,2)
hist3(Xstudent',[70 70])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Student')

% Kotz 
subplot(3,2,3)
hist3(Xkotz',[20 20])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Kotz')

% Exponential power 
subplot(3,2,4)
hist3(Xpower',[12 12])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Exp. power')

% Laplace
subplot(3,2,5)
hist3(Xlaplace',[50 50])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Laplace')

% Bessel
subplot(3,2,6)
hist3(Xbessel',[50 50])
xlim([-3 3])
ylim([-3 3])
zlim([0 200])
title('Bessel')

