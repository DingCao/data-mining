%% ======================== PCA Projecting ===============================
printf("pca running\n")
[U S] = pca(Xtrain);
best_K = 0;
for i=1:n
  if sum(sum(S(1:i, 1:i)))/sum(sum(S)) > 0.99
    best_K = i;
    break;
  endif
endfor

printf("projecting data.\n")

%Ztrain = projectData(Xtrain, U, best_K);
%Zval = projectData(Xval, U, best_K);
Ztrain = Xtrain;
Zval = Xval;
%Ztrain = [Ztrain Ztrain.^2];
%Zval = [Zval Zval.^2];

printf('Program paused. Press enter to continue.\n')
pause