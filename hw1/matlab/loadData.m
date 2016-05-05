% changes csv to mat format
printf("loading training data\n");
XAndYTrain = dlmread("data/train.csv", ",", [1, 1, 25000, 385]);
Xtest = dlmread("data/test.csv", ",", [1, 1, 25000, 384]);

% trains the fisrt four data sets with different lambda.

Ztrain = XAndYTrain(1:20000, 1:n);
ytrain = XAndYTrain(1:20000,n+1);
Zval = XAndYTrain(20001:25000, 1:n);
yval = XAndYTrain(20001:25000,n+1);

% m = Number of examples
m_train = size(Ztrain, 1);
m_val = size(Zval, 1);

%Ztrain = [Xtrain Xtrain.^2];
%Zval = [Xval Xval.^2];

clear XAndYtrain;
printf('data loaded.\n')

printf('Program paused. Press enter to continue.\n')
pause

