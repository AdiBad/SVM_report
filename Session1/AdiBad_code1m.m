cd C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1

x1 = randn(50,2)+1;
x2 = randn(51,2)-1;

y1 = ones(50,1);
y2= -ones(50,1);

disp(y2)

addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\svm')
addpath('C:\Users\Aditya\Desktop\SVM_H02D3a\Exercises\Session1\LSSVMlab')

x=-5:5;
y=-x;

figure;
hold on;
plot(x1(:,1),x1(:,2),'ro');
plot(x2(:,1),x2(:,2),'bo');
plot(x,y);
hold off;

%close
%% Initial training + testing
load('iris.mat')
% For a linear model with varying gam
gam=1;
kpar=[];

[alpha1 , b1]=svm_model(Xtrain,Ytrain,gam,kpar);

%PLOT: 
%plotlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , 'lin_kernel'}, {alpha ,b});

%% Test accuracy
rms_values=[];
gamz=[];
for i=0.001:0.005:1
    gamz=[gamz i];
    [alpha , b]=svm_model(Xtrain,Ytrain,i,kpar);
    rms_value = test_model(Xtrain,Ytrain,Xtest,Ytest,alpha,b,i,kpar);
    rms_values(end+1)= rms_value;
end
figure;
plot(1:length(rms_values),rms_values);
title('Minimum RMS at joint of curve (gamma index on X axis)');
%close;
disp('Best gam is at rms_values index: ');
min_rms = min(rms_values);
min_rms_index=min(find(rms_values==min_rms));
disp(min_rms_index);
disp(' and its value is: ');
disp(gamz(min_rms_index));

%% Automatic parameters determination
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'lin_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});

%% functions
%To get variables
function [alpha , b] = svm_model(Xtrain,Ytrain,gam,kpar)
    [alpha , b] = trainlssvm({Xtrain,Ytrain,'c',gam,kpar,'lin_kernel'});
end

% To estimate and give rms
function rms_value = test_model(Xtrain,Ytrain,Xtest,Ytest,alpha,b,gam,kpar)
    Yest = simlssvm ({ Xtrain , Ytrain , 'c', gam , kpar , 'lin_kernel'}, {alpha ,b}, Xtest);
    rms_value = rms(Yest-Ytest);
end
%%
%x=[]
%for i=0.0001:0.0001:0.05
%    x=[x i];
%end
%length(x)
