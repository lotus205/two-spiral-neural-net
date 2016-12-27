%**************************************************************************
%           CI: Twin Spirals Problem using Neural Networks 
% -------------------------------------------------------------------------
% Name: MIRANDA, Lester James Validad
% ID No: 44161652-3
% -------------------------------------------------------------------------

%% Initialization
clear; close all; clc

%% Set-up the parameters
input_layer_size = 2;
hidden_layer_size = 16;
num_labels = 2;

%% ============================= Part 0: Load two_spiral dataset from csv file ===================
% In this part, we load the two_spiral dataset from the csv file found in
% the same folder. 

fprintf('Status: Displaying data\n');
cd('C:\Users\Lj Miranda\Documents\Waseda\2016-01 Fall\110400F Computational Intelligence\reports\report-ci-final\twin-spirals')
two_spiral = csvread('two_spiral.csv');
X = two_spiral(:,1:2);  
y = two_spiral(:,3);    
gscatter(X(:,1),X(:,2),y,'rb','xo');
xlabel('x');
ylabel('y');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============================= Part 1: Initial Weight Parameters ===============================
% In this part, we initialize the weight parameters. We call the function
% randomizeInputHidden( ) and randInitializeWeights( ) in order to give
% a random initialization for both the I-H and H-O weights. It is important
% to randomize this in order to break symmetry. Afterwards, we unroll the
% initial parameters into a variable initial_nn_params

fprintf('\nStatus: Initializing Neural Network Weight Parameters\n');

initial_Theta1 = randomizeInputHidden(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============================== Part 2: Train the Neural Network ================================
% We train the neural network using different optimization techniques that
% we have built. This part is still divided into different sections, in
% order to highlight different steps in neural network training. 

fprintf('\nStatus: Training Neural Network\n');

%------------------------------------ 2.1 Training Parameters --------------------------------------
% This section puts all training parameters for all of the optimization
% techniques that we have.
trigger = 0;
% 1. Minimum Unconstrained Gradient Descent Parameters:
options_fmincg = optimset('MaxIter', 10000);
%options_fminunc = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','Display','iter');

% 2. Stochastic Gradient Descent Parameters: 
lambda = 0; 
learning_rate = 1.2;
epochs = 20000;

% 3. Particle Swarm Optimization Parameters:
epsilon = 0.1;
nvars = 82;
swarmSize = 50;
maxIter = 100;
c_1 = 1.4; % Cognitive component - exploration parameter
c_2 = 1.3; % Social component - exploitation parameter
inertiaWeight = 0.7; % Inertia Weight, smaller values makes your particles easy to converge,  
                     % Larger values makes your particles explore more. A
                     % recommended rule of thumb would be: 
                     % inertiaWeight > 0.5 * (cogEff + socEff) -1
v_initial = 0.08;

% 4. Differential Evolution Parameters:
epsilon_de = 0.1;
nvars_de = 82;
population = 10;
genMax = 100;
mutationF = 0.7; % Mutation Parameter - exploration parameter [0->2]
recombinationC = 0.01; % Recombination parameter - exploitation parameter [0->1]

%---------------------------------------------------------------------------------------------------

%-------------------------------- 2.1 Neural Network Training --------------------------------------
% In this part, we create a handle for our cost function. Thus, we have a
% costFunction that is a function that only takes one argument (the neural
% network parameters). Now, in order to make a new optimization algorithm,
% just follow these guidelines:
%       1. Let this function have the costFunction as its input argument.
%       2. The output arguments must include both the final parameters
%          nn_params, and the cost. 
%       3. Make sure you have a J_hist array to keep record of the costs
%          so that you can graph them later on.

costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% ----------------------------------------------------------------
% Optimization Algorithms (just uncomment the one you will use): 
% ----------------------------------------------------------------

% 1. Minimum Unconstrained Gradient Descent
%[nn_params, cost] = fmincg(costFunction, initial_nn_params, options_fmincg);

% 2. Particle Swarm Optimization 
getBenchmark1 = load('85_fmincg_theta1.mat');
getBenchmark2 = load('85_fmincg_theta2.mat');
benchmarkParams = [getBenchmark1.Theta1(:); getBenchmark2.Theta2(:)];
[nn_params, cost, J_hist, trigger] = particleSwarmOptimization(costFunction, benchmarkParams, epsilon, nvars, swarmSize, maxIter, c_1, c_2, inertiaWeight, v_initial);

% 3. Differential Evolution
% getBenchmark1 = load('85_fmincg_theta1.mat');
% getBenchmark2 = load('85_fmincg_theta2.mat');
% benchmarkParams = [getBenchmark1.Theta1(:); getBenchmark2.Theta2(:)];
% [nn_params, cost, J_hist, trigger] = differentialEvolution(costFunction, benchmarkParams, epsilon_de, nvars_de, population, genMax, mutationF, recombinationC);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ======================================================== Part 3: Implement Predict ====================================================================
%  After training the neural network, we would like to use it to predict
%  the labels. The "predict" function is use in the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);
accuracy = mean(double(pred == y)) * 100;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ======================================================== Part 4: Graph cost vs. iterations ====================================================================
fprintf('\nStatus: Plotting Cost History\n');
if trigger == 1
    my_iter = int16(J_hist(:,1));
    plot(my_iter,J_hist(:,2),'DisplayName','Global Best');
    title('Cost History');
    xlabel('Iterations');
    ylabel('Cost, J(\theta)');
    axis square
    hold on
    plot(my_iter,J_hist(:,3),'DisplayName','Personal Best (Mean)');
    plot(my_iter,J_hist(:,4),'DisplayName','Current Position (Mean)');
    legend('show')
    delete(findall(gcf,'Tag','particleMovement'))

elseif trigger == 2
    my_iter = int16(J_hist(:,1));
    plot(my_iter,J_hist(:,2));
    title('Cost History');
    xlabel('Generations');
    ylabel('Cost, J(\theta)');
    axis square;
end





