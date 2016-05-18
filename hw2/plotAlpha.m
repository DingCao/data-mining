function plotAlpha()
Z = dlmread("data/alpha.txt");

figure;
xlabel("iters");
ylabel("Cost");

for i=1:size(Z, 1)
    hold on;
    curve = Z(i, :);
    plot(curve, 'color',[rand,rand,rand]);
    str{i} = sprintf('alpha %f', 10^(-(i+2)));
endfor

legend(str, -1);

end
