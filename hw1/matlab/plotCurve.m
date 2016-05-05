function plotCurve(X, i)
hold on;  % don't throw the older curves

figure;

plot(X);
xlabel("iters");
ylabel("J");
legend(i);

end
