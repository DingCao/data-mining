span = 1000;

[error_train, error_val] = ...
    learningCurve(Ztrain, ytrain, Zval, yval, lambda_best, num_iters, span);

figure(2);
plot(1:m_train/span, sqrt(error_train*2), 1:m_train/span, sqrt(error_val*2));
title(sprintf('Linear Regression Learning Curve (lambda = %f)', lambda_best));
xlabel('Number of training examples');
ylabel('Error');
legend('Train', 'Cross Validation');

printf('Program paused. Press enter to continue.\n')
pause;
