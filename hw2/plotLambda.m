function plotLambda()
X = dlmread("data/lambda.txt");

Y = X(:, 2);
X = X(:, 1);

figure
%hold on;
plot(X);
xlabel("samples");
ylabel("acurrency");

figure
%hold on;
plot(Y);
xlabel("samples");
ylabel("average cost");

end
