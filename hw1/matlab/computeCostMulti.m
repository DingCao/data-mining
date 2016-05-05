function J = computeCostMulti(X, y, theta, lambda)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y



% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% CODE BY HUANG_JUNJIE@Coursera(e-mail:349373001@qq.com)

% NOTE: X matrix is without bias!
X = [ones(m,1) X];

% First we need the Hypothesis vector
h = X*theta;
% Addtion item for regularization
vecReg = [0; theta(2:length(theta))];

J = (1/(2*m)) * (h-y)'*(h-y);
% regularization for J
J = J + (lambda/(2*m)) * vecReg'*vecReg;

% =========================================================================

end
