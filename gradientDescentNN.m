function [nn_params, cost] = gradientDescentNN(costFunction, initial_nn_params, learning_rate, num_iters)
% [gradientDescentNN] Performs gradient descent to learn theta

for iter = 1 : num_iters
    [computedCost, gradient] = costFunction(initial_nn_params);
    fprintf('\nIteration %i | Cost: %j \n',iter,computedCost);
    disp(computedCost);
    initial_nn_params = initial_nn_params - learning_rate * gradient;
end

nn_params = initial_nn_params;
cost = computedCost;

end