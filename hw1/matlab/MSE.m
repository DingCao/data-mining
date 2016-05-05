function error = MSE(A, B)
% MSE compute the mean square error of two vector of same size A and B
m = size(A, 1);
error = sqrt((1/m)*(A-B)'*(A-B));

end