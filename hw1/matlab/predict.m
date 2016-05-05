% predicting the test data.

lambda_best = 0;
% retrains.
%Z = projectData([Xtrain; Xval], U, best_K);
Z = [Ztrain; Zval];
Z = [Z, Z.^2];

clear Ztrain Zval;
initial_theta = zeros(size(Z, 2)+1, 1);

[theta] = gradientDescentMulti(Z, [ytrain; yval], initial_theta,
                         alpha, num_iters, lambda_best);
J = linearRegCostFunction(Z, [ytrain; yval],
                       theta, 0); 

save "data/theta_best.mat" theta;
fprintf("totoal varience: %f, MSE: %f\n", J, sqrt(J*2))


fprintf("predicting...\n")
%Ztest = projectData(Xtest, U, best_K);
Ztest = [Xtest, Xtest .^2];
h = [ones(25000, 1) Ztest]*theta;

fprintf("saving prediction...")

fname = "data/prediction_lambda_0.1_square.csv";
f_stream = fopen(fname, "wt");
fprintf(f_stream, "%s,%s\n", "id", "reference");
fclose(f_stream);

dlmwrite(fname, [[0:24999]', h],'-append');

fprintf("Done!\n")
