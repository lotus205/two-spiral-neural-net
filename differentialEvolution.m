function [nn_params, cost, J_hist, trigger] = differentialEvolution(costFunction, initial_nn_params, epsilon, nvars, populationSize, generation, f, cr)
%DIFFERENTIALEVOLUTION Finds the optimal solution of an objective
% function using the Differential Evolution algorithm.
%
% DE Steps: 
% 1. Initialization - initialize particles.
% 2. Mutation - randomly select three vectors and compute for trial vector.
% 3. Recombination - from a donor vector, do some exchange of genes using
%                     the probability C.
% 4. Selection - do tournament selection.



%% =============================== Initialize Parameters ===================================
% Particle Vector 
particleVec = [];
for iter = 1:populationSize
    temp = epsilon.* randn(nvars,1) + initial_nn_params;
    particleVec(iter,:) = vertcat(temp);
end

% Cost Tracker
J = [];

% Cost History
J_hist = [];

for iter = 1:generation
%% =============================== Mutation ================================================
% For mutation, you choose three random vectors, and relate them in order
% to build the donor vector (donorVec). This method is controlled by the
% parameter f. 
    
    randIndex = randperm(populationSize,3);
    donorVec = particleVec(randIndex(3),:) + f * (particleVec(randIndex(1),:) - particleVec(randIndex(2),:));
    
%% ============================ Recombination ==============================================
% For recombination, you build your trial vector (trialVec) using the
% elements of the donor vector (donorVec) and the elements of your target vector.
% This means that for each particle in the population (for each row of
% particle) and for each dimension of that particle (for each column of
% each row of particle), you have to do the comparison.
    
    trialVec = zeros(populationSize,nvars);
    randomNumVec = rand(populationSize,nvars);
    i_rand = randperm(nvars,1);
    for particleIndex = 1:size(particleVec,1)
        for dimensionIndex = 1:nvars
            if randomNumVec(particleIndex,dimensionIndex) <= cr || dimensionIndex == i_rand
                trialVec(particleIndex,dimensionIndex) = donorVec(1,dimensionIndex);
            elseif randomNumVec(particleIndex,dimensionIndex) > cr && dimensionIndex ~= i_rand
                trialVec(particleIndex,dimensionIndex) = particleVec(particleIndex,dimensionIndex);
            end
        end
    end

%% ============================ Selection ==============================================
% For selection, the target vector (particle) is then compared against the
% trial vector (trialVec) if costFunction(trialVec(particleIndex,:)) is less than the computed
% costFunction(particleVec(particleIndex,:)), then you just replace that
% particle with the one captured from trialVec. After all particles are
% tested, we then have the second generation.
    
    for particleIndex = 1:size(particleVec,1)
        if costFunction(trialVec(particleIndex,:)') <= costFunction(particleVec(particleIndex,:)')
            particleVec(particleIndex,:) = trialVec(particleIndex,:);
        end
    end

%% ============================ Get Fittest Gene for this Generation =========================================
    for particleIndex = 1:size(particleVec,1)
        J(particleIndex) = costFunction(particleVec(particleIndex,:)');
    end
    
    [minimaJ,  minimaJIndex] = min(J);
    J_hist = [J_hist' [iter; minimaJ]]';


    Y = ['Generation ',num2str(iter),'| Fittest Gene Cost: ', num2str(minimaJ)];
    disp(Y);
    
end

nn_params = particleVec(minimaJIndex,:)';
cost = min(J);
trigger = 2;

end

