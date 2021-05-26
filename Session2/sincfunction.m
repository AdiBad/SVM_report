%Using LSSVM regression
X = (-3:0.2:3)';
Y = sinc(X)+0.1.*randn(length(X),1);
gam = 1000;
sig2 = 0.3;
type = 'function estimation';

%Xt = 3.*randn(10,1);
%Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);

[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','original'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'});

%% Actual code starts here:
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\svm')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\LSSVMlab')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\export_fig')

X = (-3:0.2:3)';
Y = sinc(X)+0.1.*randn(length(X),1);

Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);
gam = 1000000;
sig2 = 100;
type = 'function estimation';

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'});
Yt = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);

plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on; 
plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'r-.');
hold off; 
legend('true','Yt','YtEst');

err = immse(Yt,Ytest); 
err