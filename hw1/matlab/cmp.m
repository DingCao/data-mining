best = dlmread("data/prediction_best_8.07.csv", ",", [1, 1, 25000, 1]);
current = dlmread("data/prediction_lambda_0.01_2.csv", ",", [1, 1, 25000, 1]);

printf("difference: %f\n", MSE(best, current))
