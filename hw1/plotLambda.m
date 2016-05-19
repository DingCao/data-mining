function plotLambda()
Z = dlmread("data/lambda.txt");

error_train = Z(:, 1);
error_val = Z(:, 2);

figure
xlabel("lambda");
ylabel("Cost");

hold on;
plot(error_train, 'color', 'red');
plot(error_val, 'color', 'green');
legend('train set', 'validation set');

end
