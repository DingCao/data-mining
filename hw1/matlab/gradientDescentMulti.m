function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % CODE BY HUANG_JUNJIE@Coursera(e-mail:349373001@qq.com)
    
    % d for the derivation vector of J.
    % first compute the hypothesis vector h = X * theta,
    % then compute the h-y which is also a vector.
    % finally the derivation vector will be 1/m * ((h-y)'*X)'
    [J_history(iter) grad] = linearRegCostFunction(X, y, theta, lambda);
    
    % now we get the vectorization of derivations, we can update the theta with
    % just a linear computation.
    theta = theta - alpha * grad;
    %fprintf("iter: %8d, cost: %f\r", iter, J_history(iter));

    % ============================================================
end

end
