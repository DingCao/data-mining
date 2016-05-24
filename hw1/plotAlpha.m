function plotAlpha()
ALPHAS = [3e-2, 1e-2, 3e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6];
Z = dlmread("data/alpha.txt");

figure;
xlabel("iters/20");
ylabel("Cost");

for i=1:size(Z, 2)
    hold on;
    curve = Z(:, i);
    plot(curve, 'color',[rand,rand,rand]);
    str{i} = sprintf('alpha %f', ALPHAS(1, i));
endfor

legend(str, -1);

end
