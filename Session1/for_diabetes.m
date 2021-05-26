load diabetes.mat
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\svm')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\LSSVMlab')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\export_fig')

X=trainset; Y=labels_train; Xt=testset; Yt=labels_test;
Xtrain=X;
Ytrain=Y;
%% data visualization 
scatter([1:8],total)

%%
% Linear model

type='c';

[gam ,sig2 , cost ] = tunelssvm ({ X , Y , 'c', [], [],'lin_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
[alpha,b] = trainlssvm({Xt,Yt,type,gam,[],'lin_kernel'});

figure('Color',[1 1 1]);
%subplot(1,2,1);
plotlssvm({Xt,Yt,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
%hold on;
[Yht, Ylin] = simlssvm({Xt,Yt,type,gam,[],'lin_kernel'}, {alpha,b}, X);

%export_fig('ripley_linear.pdf')
%%
%Polynomial model

type='c'; 

t = 1; 
degree = 5;

%Use tunnel to detect gam
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [t; degree],'poly_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});

disp('Polynomial kernel of degree 5'),

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});
figure; 
plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
[Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);

err = sum(Yht~=Ytest); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Ytest)*100)

%%
% RBF kernel tuning 
model_csa = {Xt, Yt, 'c', [], [], 'RBF_kernel', 'csa'};
[gam, sig2, cost] = tunelssvm(model_csa, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});

[alpha,b] = trainlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'});
[estYval, YRBF] = simlssvm({Xt,Yt,'c',gam,sig2,'RBF_kernel'}, {alpha,b}, X);
err = sum(estYval~=Y);
figure; 
plotlssvm({Xt,Yt,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

%%
% ROC curve
figure('Color',[1 1 1]);
roc(Ylin, Y);

figure('Color',[1 1 1]);
roc(Zt, Yt);

figure('Color',[1 1 1]);
roc(YRBF, Y);

export_fig('ripley_ROC.pdf')
