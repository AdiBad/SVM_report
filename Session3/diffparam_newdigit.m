%
% Experiments on the handwriting data set on kPCA for reconstruction and denoising
% switch Xtest1 with Xtest2 for testing different dataset, can change Xdt
% with Xdt_lin to get results for linear kernel

load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest2,1);
minx=min(min(X)); 
maxx=max(max(X));

noisefactor =1;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X; 
for i=1:N;
  randn('state', i);
  Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest2; 
for i=1:size(Xtest2,1);
  randn('state', N+i);
  Xnt(i,:) = Xtest2(i,:) + noise*randn(1,dim);
end

%
% select training set
%
Xtr = X(1:1:end,:);



sig2 =dim*mean(var(Xtr)); % rule of thumb

sigmafactor = 10;

sig2=sig2*sigmafactor;




%
% kernel based Principal Component Analysis using the original training data
%


disp('Kernel PCA: extract the principal eigenvectors in feature space');
disp(['sig2 = ', num2str(sig2)]);


% linear PCA
[lam_lin,U_lin] = pca(Xtr);

% kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240); 
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

npcs = [2.^(0:7) 190];
siglist = logspace(-2,2,9)
%number of principal components to extract
lpcs = length(npcs); %length of tried principal components
errors = [];
error_list = ones(1,9);
% choose the digits for test
digs=[0:9]; ndig=length(digs);
m=2; % Choose the mth data for each digit 

Xdt=zeros(ndig,dim);

for k=1:lpcs
    for sig = siglist 
    MSE = 0;
    nb_pcs=npcs(k);
    Ud=U(:,(1:nb_pcs)); % U is the eigenvectors of the kernel PCA
    for i=1:ndig %
        noised_digit=Xnt(i,:);
        denoised_digit(i,:) = preimage_rbf(Xtr,sig2,Ud,noised_digit,'denoise');
        MSE = MSE + sum((denoised_digit(i,:) - Xtest2(i,:)).^2);
    end
reconstruction_error = MSE/ndig;
errors = [errors, reconstruction_error];
    end
    error_list = [error_list; errors];
    errors = [];
end

figure('Color',[1 1 1]);
surf(npcs, siglist, error_list(2:10,:),'FaceAlpha',0.5);
set(gca,'Xdir','reverse','Ydir','reverse', 'XScale', 'lin', 'YScale', 'log')
title('Validation set model performance plotted on npcs and Sigma');
xlabel('Npcs' ); ylabel('Sigma^2 (log)'); zlabel('MSE');