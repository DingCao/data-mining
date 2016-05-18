function plotValidation()
Z = dlmread("data/validated.txt");

error_train = Z(:, 1);
error_val = Z(:, 2);

figure
xlabel("m*100");
ylabel("Cost");

hold on;
plot(X, 'color', 'red');
plot(Y, 'color', 'green');
legend('train set', 'validation set');

end
