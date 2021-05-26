load('santafe.mat')


%% figure data


figure;
plot(1:length(Z),Z);
title('Z values');
%% Tune the order
% first attempt
order=50;
X=windowize(Z,1:(order+1));
Y=X(:,end); X=X(:,1:order);
[gam,sig2]=tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
inputs = bay_lssvmARD({X,Y,'f',gam,sig2,'RBF_kernel'});
%save('inputsARD', 'inputs');

%%

order = 50;
X = Z(:, 1);
Xt = Ztest(:, 1);
Xu = windowize(Z,1:order+1);
Xtra = Xu(1:end-order,1:order);

%training set
Ytra = Xu(1:end-order,end);
Xs=Z(end-order+1:end,1);

% tuning of model parameter
%[gam, sig2]=tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
% ARD
inputs = bay_lssvmARD({Xu(:,1:order),Xu(:,end),'f',gam,sig2,'RBF_kernel'});
%inputs = [16  18  21  22  23  24];
%prediction
prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,200);

%%
figure('Color', [1 1 1]);
plot([prediction, Xt(1:200)]);
legend('prediction', 'Test data');
mae([prediction, Xt(1:200)])
immse(prediction, Xt(1:200))