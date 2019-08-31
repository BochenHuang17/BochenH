function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X = [ones(m,1), X]; %5000x401

yk = zeros(m,num_labels);
for i =1:m
a1 = X(i,:)';
z2 = Theta1*a1; % 25x1
a2 = [1; sigmoid(z2)]; %26x5000
z3 = Theta2*a2; %10x1
a3 = sigmoid(z3);% 10x1
hypo= a3; %这个example的输出为hypo，是一个10维向量
yk(i,y(i))=1;
temp_y = yk(i,:)'; % 10x1
temp1 = -temp_y.*log(hypo);
temp2 = (1-temp_y).*log(1-hypo);
J_temp = sum(temp1-temp2); 
J =J+J_temp;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function. //got it! by setting temp_y
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


%computer the error of units in layer3(output layer):
delta3 = hypo-temp_y;% 10x1
%computer the error of units in layer2(hidden layer):
delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2); % 25x1 note that you should skip or remove delta_sup(2)_sub0, which corresponds to the bias term.
%accumulating from this example:
Theta1_grad = Theta1_grad+delta2*a1'; %25x401
Theta2_grad = Theta2_grad+delta3*a2'; %10%26

%Del2 should be the same size of Theta2, while Del1 should be the same size of Theta1
%Del2 = Del2+ delta3*a2';%10x26
%dont forget to delta_super(2)_sub(0),which is the 'error'of bias term
%Del1 = Del1+ delta2*a1'; %25x401
 
endfor

%dont forget to *(1/m)

Theta1(:,1)=0; %disclude bias terms first colume = 0
Theta2(:,1)=0; %disclude bias terms first colume = 0
Theta1_grad = (1/m)*Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = (1/m)*Theta2_grad + (lambda/m)*Theta2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

r=(lambda/m/2)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J = (1/m)*J+r;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
