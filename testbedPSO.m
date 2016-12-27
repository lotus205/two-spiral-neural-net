%**************************************************************************
%           CI: PSO Testbed 
% -------------------------------------------------------------------------
% Name: MIRANDA, Lester James Validad
% ID No: 44161652-3
% -------------------------------------------------------------------------
clear; close all; clc

% Set-up the parameters
input_layer_size = 2;
hidden_layer_size = 16;
num_labels = 2;

fprintf('Status: Displaying data\n');
cd('C:\Users\Lj Miranda\Documents\Waseda\2016-01 Fall\110400F Computational Intelligence\reports\report-ci-final\twin-spirals')
two_spiral = csvread('two_spiral.csv');
X = two_spiral(:,1:2);  
y = two_spiral(:,3);    


%% ============================== TestBed ================================
fprintf('\nStatus: Running Testbed\n');
% Initialize params:
getBenchmark1 = load('85_fmincg_theta1.mat');
getBenchmark2 = load('85_fmincg_theta2.mat');
benchmarkParams = [getBenchmark1.Theta1(:); getBenchmark2.Theta2(:)];

% Particle Swarm Optimization Parameters:
epsilon = 0.1;
nvars = 82;
swarmSize = 50;
maxIter = 100;
%inertiaWeight = 0.7; 
v_initial = 0.08;
lambda = 0;
c1 = 1.5;
c2 = 1.5;
% Range for c1 and c2:
dx = 0.1;
inertia_range = (0:dx:1.0); 


% Make costFunction Handle
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);

% cmatrix
%cmatrix = [];

% wmatrix
wmatrix = [];


% myIter
myIter = 0;

%% ======================================== Testbed for c1 and c2 params ====================================
% for i = 1:size(c1_range,2)
%     for j = 1:size(c2_range,2)
%         myIter = myIter + 1;
%         G = ['Running Testbed PSO with c1: ',num2str(c1_range(i)),' and c2: ',num2str(c2_range(j))];
%         disp(G);
%         [nn_params, cost] = particleSwarmOptimization(costFunction, benchmarkParams, epsilon, nvars, swarmSize, maxIter, c1_range(i), c2_range(j), inertiaWeight, v_initial);
%         Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
%         Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
%         pred = predict(Theta1, Theta2, X);
%         accuracy = mean(double(pred == y)) * 100;
%         Y = ['Accuracy: ', num2str(accuracy)];
%         disp(Y);
%         cmatrix = [cmatrix' [c1_range(i); c1_range(j); accuracy]]';
%         if mod(myIter,10)== 0
%             disp(cmatrix);
%             fprintf('Program paused. Press enter to continue.\n');
%             pause;
%         end
%     end
% end

%% ====================================== Testbed for inertiaweight param ========================================

for i = 1:size(inertia_range,2)
    G = ['Running Testbed PSO with w: ',num2str(inertia_range(i))];
    disp(G);
    [nn_params, cost] = particleSwarmOptimization(costFunction, benchmarkParams, epsilon, nvars, swarmSize, maxIter, c1, c2, inertia_range(i), v_initial);
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
    pred = predict(Theta1, Theta2, X);
    accuracy = mean(double(pred == y)) * 100;
    Y = ['Accuracy: ', num2str(accuracy)];
    disp(Y);
    wmatrix = [wmatrix' [inertia_range(i); accuracy]]';
end

disp(wmatrix);
fprintf('Program paused. Press enter to continue.\n');
pause;
