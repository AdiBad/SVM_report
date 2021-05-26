load iris


%
% train LS-SVM classifier with linear kernel 
%
type='c'; 
gam = 1; 
disp('Linear kernel'),
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\svm')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\LSSVMlab')

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

disp('Press any key to continue...'), pause, 
%%



%
% Train the LS-SVM classifier using polynomial kernel
%
type='c'; 
gam = 1; 
t = 1; 
degree = 10;
disp('Polynomial kernel of degree 5'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)
%disp('Press any key to continue...'), pause,        
 %%   

%
% use RBF kernel
%

% tune the sig2 while fix gam
%
disp('RBF kernel')
gam = 1; sig2list=[0.01, 0.1, 1, 5, 10, 25];

errlist=[];

for sig2=sig2list,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end

%%

% tune the gam while fix sig2
%
disp('RBF kernel')
sig2 = 0.1; gamlist=[0.01, 0.1, 1, 10];

errlist=[];

for gam=gamlist,
    disp(['gam : ', num2str(gam), '   sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end

%%
%Performance with different tuning parameters for training and validation 

%change below gam and sig
gam = 0.01;
sig2 = 0.01;

perf1 = [];
perf2 = [];
perf3 = [];

gamlist = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
siglist = [0.001, 0.01, 0.1, 1, 10, 100, 1000];

for sig2 = siglist,
    %Random split
    perf1 = [perf1 rsplitvalidate({Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 0.80 , 'misclass')];

    % k-fold crossvalidation
    perf2 = [perf2 crossvalidate({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 10, 'misclass')];

    % Leave-one-out validation
    perf3 = [perf3 leaveoneout({ Xtrain , Ytrain , 'c', gam , sig2 ,'RBF_kernel'}, 'misclass')];
end

figure;
hold on;
plot(gamlist, perf1,'r');
plot(gamlist, perf2,'g');
plot(gamlist, perf3,'b');
hold off;
title('Test performance sigma (fixed gamma^2=0.01)');
xlabel('sigma'); ylabel('Estimated cost');legend('rsplit','xvalidate','loocv');

disp([gam, sig2, perf1, perf2, perf3]);
%%
%Try different algorithms in automatic parameter tuning

%[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
run1_simplex = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
run2_simplex = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
run3_simplex = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});

run1_grid = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
run2_grid = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
run3_grid = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});

disp([run1_simplex run2_simplex run3_simplex]);
disp([run1_grid run2_grid run3_grid]);

%%
%Bayesian framework probability estimates
gam=11;
sig2=0.15;

bay_modoutClass({Xtrain , Ytrain , 'c', gam , sig2}, 'figure');

%%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclass'),



