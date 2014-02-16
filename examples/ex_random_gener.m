% This example shows how to generate spherically symmetric random vectors.

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

