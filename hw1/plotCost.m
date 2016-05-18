function plotCost()
Z = dlmread("data/cost.txt");

#Y = Z(2, :);
X = Z(:, 1);

figure
xlabel("m*100");
ylabel("Cost");

hold on;
plot(X, 'color', 'red');
#plot(Y, 'color', 'green');
legend('train set');

end
