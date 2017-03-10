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
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part1
X = [ones(m,1) X]; % add the theta 0 line
h2 = Theta1 * X' ; % 25×401 * 401xm = 25xm
a2 = sigmoid(h2) ;
a2 = [ones(m,1) a2']; % mx26

h3 = Theta2 * a2'; %10×26 * 26xm = 10xm
a3 = sigmoid(h3);
Y = zeros(num_labels,m);

 

%Part2
for index = 1:m
	row = y(index);
	Y(row, index) = 1;
end

J = 1/m * sum(sum( -Y .* log(a3) - (1-Y) .* log(1 - a3))) ;

J = J + lambda/(2*m) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );

for i = 1:m
	a1 = X(i,:); % 1×401

	z2 = Theta1 * a1'; % 25×401 * 401×1 = 25×1 vector
	a2 = sigmoid(z2);
	a2 = [1; a2]; % add bias unit 26×1 vector

	z3 = Theta2 * a2; % 10×26 * 26×1 = 10×1
	a3 = sigmoid(z3); %

	d3 = a3 - Y(:,i); % 10×1
	d2 = Theta2'*d3 .* sigmoidGradient([1;z2]); % (26×10 * 10×1) .* 26×1 = 26×1 .* 26×1

	% Initially I implemented wrongly for the backprop algorithm
	% My mistake was that I wrote d2 = Theta2’*d3 .* sigmoidGradient(a2)
	% What I was computing is actually sigmoidGradient of the sigmoid of z2 –> sigmoidGradient(sigmoid(z2))
	% What we needed was computing the sigmoidGradient(z2)
	% Also I need to add a bias unit to z2 ( [1;z2] ) to make it a 26×1 vector
	Theta1_grad = Theta1_grad + d2(2:end) * a1; % 25×1 * 1×401 = 25×401
	Theta2_grad = Theta2_grad + d3 * a2'; % 10×1 * 1×26 = 10×26
end

 

%Part3
Theta1_grad(:,1:1) = 1/m * Theta1_grad(:,1:1);
Theta2_grad(:,1:1) = 1/m * Theta2_grad(:,1:1);

Theta1_grad(:,2:end) = 1/m * ( Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) ) ;
Theta2_grad(:,2:end) = 1/m * ( Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) ) ;

% I also make mistake in calculating Theta1 and Theta with regularization
% Below is my initial implementation which did not pass the test

% Theta1_grad = 1/m * Theta1_grad;
% Theta2_grad = 1/m * Theta2_grad;

% Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1_grad(:,2:end)
% Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2_grad(:,2:end)

% Let take only one element (rOriginal) from theta1_grad and put into simpler terms
% wrong intepretation
% rtemp = rOriginal/m
% rfinal = rtemp + l*rtemp/m

% correct implementation
% rfinal = (rOriginal + l*rOriginal) / m = rOriginal/m + l*rOriginal/m (can be too written as below)
% = rtemp + l*rOriginal/m

% the subtle difference is l*rtemp vs l*rOriginal where rtemp != rOriginal therefore I got it wrong initially


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
