function plotCost()
Z = dlmread("data/cost.txt");

Y = Z(:, 2);
X = Z(:, 1);

figure
xlabel("Iter");
ylabel("Current Cost");
legend(1);
%hold on;
plot(X);


%figure
hold on;
plot(Y);
ylabel("Average Cost");
legend(2);

end
