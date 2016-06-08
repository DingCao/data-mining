function plotCost()
Z = dlmread("data/cost.txt");

X = Z(:, 1);

figure
hold on
xlabel("Iter*10");
ylabel("Current Cost");
plot(X, 'color', 'red');
legend("train set")

end
