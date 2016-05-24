function plotCost()
Z = dlmread("data/cost.txt");

X = Z(:, 1);

figure
xlabel("Iter");
ylabel("Current Cost");
legend(1);
plot(X, 'color', 'red');

end
