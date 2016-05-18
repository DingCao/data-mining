function plotValidation()
Z = dlmread("data/validated.txt");

Y = Z(:, 2);
X = Z(:, 1);

figure
xlabel("m");
ylabel("Cost");

hold on;
plot(X, 'color', 'red');
plot(Y, 'color', 'green');
legend('train set', 'validation set');

end
