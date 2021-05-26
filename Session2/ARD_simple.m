%% Using bayesian framework to do ARD
sig2 = 0.4;
gam = 0.01;
X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({X, Y, 'f', gam , sig2 });