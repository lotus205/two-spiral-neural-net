function [J, grad] = nnCostFunction(nn_params, ...
                                    input_layer_size, ...
                                    hidden_layer_size, ...
                                    num_labels, ...
                                    X, y, lambda)
                                
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%
% Parameters:
% nn_params - for first-use, put the initial neural network parameters you
%             have
% input_layer_size - number of nodes in the input layer
% hidden_layer_size - number of nodes in the hidden layer
% num_labels - the number of categories you have
% X - input variables
% y - target variables
% lambda - regularization parameter.

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));



%% ============================================== FEEDFORWARD IMPLEMENTATION =========================================

a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = sin(z_2);
a_2 = [ones(size(a_2,1), 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
hypothesis = a_3;


y_i = zeros(m,num_labels);
for i = 1:m
    y_i(i,y(i)) = 1;
end

J = 1/m * sum(sum(-1 * y_i .* log(hypothesis)-(1-y_i) .* log(1-hypothesis)));
% J = 1/m * sumsqr(hypothesis - y_i);
%J = (1/m) * sum(sum(-1 * ((y_i + 1) / 2) .* log((hypothesis + 1) / 2) - (1 - (y_i / 2)) .* log(1 - (hypothesis + 1) / 2)));
regularization_term = (lambda / (2 * m) * (sum(sumsqr(Theta1(:,2:input_layer_size+1))) + sum(sumsqr(Theta2(:,2:hidden_layer_size+1)))));
J = J + regularization_term;

%% ============================================== BACKPROPAGATION IMPLEMENTATION =========================================

for t = 1:m
	a_1 = [1; X(t,:)'];      % Attach bias vector (with ones) on a1
	z_2 = Theta1 * a_1;       % Compute pre-activation z2
	a_2 = [1; sin(z_2)];     % Compute activation a2 (using tanh), attach bias vector
	z_3 = Theta2 * a_2;       % Compute pre-activation z3
	a_3 = sigmoid(z_3);       % Compute activation a3 (using sigmoid)
	target_output = ([1:num_labels]==y(t))';

	delta_3 = a_3 - target_output;   % Compute the difference between the activation result and the target output
	delta_2 = (Theta2' * delta_3) .* [1; cos(z_2)]; % Compute delta_2 , always get the g' of layer 2. 
	delta_2 = delta_2(2:end);

	% Accumulate Delta
	Theta1_grad = Theta1_grad + delta_2 * a_1';
	Theta2_grad = Theta2_grad + delta_3 * a_2';
end

% Compute for the partial derivative terms
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

grad = [Theta1_grad(:) ; Theta2_grad(:)];
% ======================================================================================================================


end






















