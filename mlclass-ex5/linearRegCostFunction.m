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


Xt = X * theta;
J = 1/(2*m) * (Xt - y)' * (Xt - y);

reg = 0;
n = length(theta);
for(j = 2 : n)
   reg += theta(j)^2;
end;
reg *= lambda/(2*m);
J += reg;





% =========================================================================

for(i = 1 : m)
   z = theta' * X(i,:)';
   err = z - y(i);
   partgrad = err * X(i,:)';
   grad += partgrad;
end;
grad /= m;

regGrad = theta * lambda/m;
regGrad(1) = 0;

grad += regGrad;

grad = grad(:);

end
