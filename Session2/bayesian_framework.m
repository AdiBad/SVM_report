sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
%The model can be optimized with respect to these criteria:
[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gam] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
%For regression, the error bars can be computed using Bayesian inference using
sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');