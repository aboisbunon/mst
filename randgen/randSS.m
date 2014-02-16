% Function generating spherically symmetric random vectors 

% Input: 
%   N = size of the random vectors we wish to generate
%   M = number of random vectors to generate
%   options = structure with the following elements:
%       -   options.type = name of the distribution ('gauss' for Gaussian,
%       't_mg' or 't_fisher' for Student, 'kotz' for Kotz, 'mg' for Gaussian 
%       mixture, 'exp_power' for Exponential power, 'laplace' for multivariate
%       spherical Laplace, 'unifSS' for uniform on sphere, 'pearsonII' for 
%       Pearson type II.
%       -   other elements of options correspond to parameters of each law.
%
% Output: 
%   X = matrix of size NxM containing the M random vectors in column.
%
% By A. Boisbunon, April 2011.
% Modified in August 2012.
% Modified in November 2012.

% TODO: change random by gamrnd and others (random not supported by Octave)


 
function [X] = randSS(N, M, options)

    if ~isfield(options,'type')
        options.type='gauss';
    end
    if ~isfield(options,'scale')
        options.scale=1;
    end
    if ~isfield(options,'df')
        options.df=100;
    end
    if ~isfield(options,'rate')
        options.rate=1;
    end
    if ~isfield(options,'power')
        options.power=1;
    end
    if ~isfield(options,'N_kotz')
        options.N_kotz=2;
    end    
    if ~isfield(options,'r')
        options.r=.5;
    end
    if ~isfield(options,'s')
        options.s=1;
    end
    if ~isfield(options,'shape')
        options.shape=1;
    end    
    if ~isfield(options,'q')
        options.q=1;
    end    
    if ~isfield(options,'p2')
        options.p2=1;
    end    
    

    switch options.type
        case 't_fisher' % multivariate student (with Fisher radius)
            R = sqrt(N*random('f', N, options.df, M, 1));
            bool_rayon = 1 ;
        case 't_mg'    % multivariate student via gaussian mixture
            v = 1./random('gam', options.df/2, 2/options.df, M, 1);
            X = randn(N,M).*repmat(sqrt(v'),N,1);
            bool_rayon = 0 ;
        case 'exp'  % exponential radius
            R = exprnd(options.rate, M, 1);
            bool_rayon = 1 ;
        case 'wbl'  % weibull radius
            R = random('wbl', options.scale, options.shape, M, 1);  
            bool_rayon = 1 ;
        case 'kotz' % Multivariate Kotz with gamma radius
            R = sqrt(random('gam', options.N_kotz+N/2+1,options.r/options.s^2,M,1));
            bool_rayon = 1 ;
        case 'exp_power'% Multivariate exponential power with gamma radius
            R = random('gam', N/(2*options.power),2*options.scale^(2*options.power),M,1).^(1/(options.power*2));
            bool_rayon = 1 ;
        case 'ep_kotz'% Multivariate exponential power through special case of Kotz
            R = sqrt(random('gam', N/2+2,options.power/options.scale^2,M,1));
            bool_rayon = 1 ;
        case 'laplace' % Multivariate Laplace through Gaussian mixture
            v = random('gam',N/2,1/options.scale^2,M,1) ;
            bool_rayon = 0 ;
            X = randn(N,M).*repmat(sqrt(v'),N,1);
        case 'bessel'
            v = random('gam',options.q+N/2,1/(2*options.scale^2*options.r),M,1) ;
            bool_rayon = 0 ;
            X = randn(N,M).*repmat(sqrt(v'),N,1);            
        case 'unifSS' % Uniform distribution on a sphere of radius R
            R = repmat(options.scale,M,1) ;
            bool_rayon = 1 ;
        case 'pearsonII'
            R = sqrt(random('beta', N/2,options.p2,M,1)) ;
            bool_rayon = 1 ;
        otherwise % Gaussian distribution with chi2 distribution
            R = sqrt(random('chi2', N, M, 1)) ;    
            bool_rayon = 1 ;
    end

    % If what we have generated sofar corresponds to the radius of the
    % vector, we now have to generate its direction.
    if (bool_rayon == 1)
        Y = randn(N,M) ;
        normY = sqrt(diag(Y'*Y)) ;
        matNormY = repmat(normY',N,1);
        U = Y./matNormY;
        X = U.*repmat(R',N,1);
    end
            
end

