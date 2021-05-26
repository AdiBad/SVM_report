load logmap.mat
%Two variables are loaded into the workspace: Z (training data) and Ztest (test data).
%First, we have to map our sequence Z into a regression problem. This can be done using the
%command windowize:
 order = 20;
 X = windowize (Z, 1:( order + 1));
 Y = X(:, end);
 X = X(:, 1: order );
%Now, a model can be built using these data points:
 gam = 1000000;
 sig2 = 1000;
 
 [alpha , b] = trainlssvm ({X, Y, 'f', gam , sig2 });
%It is straightforward to predict the next data points using the predict function of the LSSVMlab toolbox. In order to call the function, we first have to define the starting point of
%the prediction:
Xs = Z(end - order +1: end , 1);

%Naturally, this is the last point of the training set. The test set Ztest presents data points
%after this point, which we will try to predict. This can be implemented as follows:
 nb = 50;
 prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
%% where nb indicates how many time points we want to predict. Here, we define this number
%equal to the number of data points in the test set. Finally, the performance of the predictor
%can be checked visually:
 figure ;
 hold on;
 plot (Ztest , 'k');
 plot (prediction , 'r');
 hold off
 legend('Ztest','pred');