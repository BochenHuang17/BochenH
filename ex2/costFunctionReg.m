function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
id=eye(length(theta));
id(1,1)=0;

z = X*theta;
hypo = sigmoid(z);
temp1 = -y.*log(hypo);
temp2 = (1-y).*log(1-hypo);
J = (1/m)*sum(temp1-temp2)+(lambda/(2*m))*sum(id*theta.*theta);
grad = (1/m)*(X'*(hypo -y))+(lambda/m)*(id*theta);

% =============================================================

end
