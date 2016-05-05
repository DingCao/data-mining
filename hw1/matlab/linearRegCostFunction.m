function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% CODED BY HUANG_JUNJIE@COURSERA(E-MAIL: 349373001@QQ.COM)
% DATE: 2016-2-5

% NOTE: X matrix is without bias!
X = [ones(m,1) X];

% First we need the Hypothesis vector
h = X*theta;
% Addtion item for regularization
vecReg = [0; theta(2:length(theta))];

J = (1/(2*m)) * (h-y)'*(h-y);
% regularization for J
J = J + (lambda/(2*m)) * vecReg'*vecReg;

% Compute the grad as vector
grad = (1/m) * X'*(h-y);
% regularization for grad
grad = grad + (lambda/m) * vecReg;

% =========================================================================

grad = grad(:);

end
