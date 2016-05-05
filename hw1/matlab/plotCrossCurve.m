printf("choosing lambda\n")

lambda_best = 0;

[lambda_vec, error_train, error_val] = ...
    validationCurve(Ztrain, ytrain, Zval, yval, lambda_vec, num_iters);
[min_val_error, index_min_val_error] = min(error_val);
lambda_best = lambda_vec(index_min_val_error);

printf("best lambda: %f\n", lambda_best);

plot(1:length(lambda_vec), sqrt(error_train*2), 1:length(lambda_vec), sqrt(error_val*2));
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

printf('Program paused. Press enter to continue.\n')
pause
