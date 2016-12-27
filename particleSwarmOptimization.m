function [nn_params, cost, J_hist, trigger] = particleSwarmOptimization(costFunction, initial_nn_params, epsilon, nvars, swarmSize, maxIter, c_1, c_2, inertia, v_initial)
% PARTICLESWARMOPTIMIZATION Finds the optimal solution of an objective
% function using the PSO optimization algorithm.
% 
%
% Parameters:
% epsilon - defines the spread of the particles in the search area.
% nvars - defines the number of parameters to be minimized in the objective
%         function
% swarmSize - defines the number of particles found in the swarm.
% maxIter - number of iterations for the algorithm to run.
% c_1 - cognitive coefficient (exploration parameter)
% c_2 - social coefficient (exploitation parameter)
% inertiaWeight - defines the inertia for the particle movement.
% v_initial - sets the initial nudge of the particles. 

%% =================== Initialize Different Matrices =======================
% currentPos
currentPos = [];
for iter = 1:swarmSize
tempSwarm = epsilon.* randn(82,1) + initial_nn_params;
currentPos(iter,:) = vertcat(tempSwarm);
end

% pbestPos
pbestPos = currentPos;

% gbestPos
for particle = 1:swarmSize
    temp = currentPos(particle,:)';                    
    [J] = costFunction(temp);                       
    jMat(particle,:) = vertcat(J);  
end
[~, minIndex] = min(jMat); 
gbestPos = currentPos(minIndex,:); 


% velocityMatrix
velocityMatrix = v_initial * ones(swarmSize,nvars);

% J_hist and other graphing vectors
J_hist = [];
pBestJVec = [];
currentPosJVec = [];


%% ========================================= Actual PSO ====================================================
filename = 'pso_finalrun_1.gif';
for iter = 1:maxIter
    for particle = 1:swarmSize        
        % Set the personal best option
        currentPosJVec = [currentPosJVec costFunction(currentPos(particle,:))];
        if costFunction(currentPos(particle,:)') < costFunction(pbestPos(particle,:)')
            pbestPos(particle,:) = currentPos(particle,:);
        end
        pBestJVec = [pBestJVec costFunction(pbestPos(particle,:))];
        % Set the global best option
        if costFunction(currentPos(particle,:)') < costFunction(gbestPos')
             gbestPos = currentPos(particle,:);
        end    
    end

%-------------------------------------------------------------------------------------------------------------------------------------    
p = 1;
myCurrent = costFunction(currentPos(p,:)');
myBest = costFunction(pbestPos(p,:)');
ourBest = costFunction(gbestPos');
X = ['Iteration ',num2str(iter),'| currentPos: ', num2str(myCurrent), ' | pbestPos: ', num2str(myBest), ' | gbestPos: ', num2str(ourBest)];
disp(X);



scatter(currentPos(:,1),currentPos(:,2), 'MarkerEdgeColor','k','MarkerFaceColor',[1 1 0]);
xlim([7.0 7.4]);
ylim([6.1 6.5]);
xlabel('\theta_{11}');ylabel('\theta_{12}');
axis square;
title(['Plot of Particle Movement']);
dim = [.55 .30 .6 .6];
str = {'Parameters',strcat('Swarm size: ', num2str(swarmSize)), strcat('Particle spread: ', num2str(epsilon)), strcat('Inertia: ', num2str(inertia)), strcat('c: ', num2str(c_1)), strcat('s: ',num2str(c_2))};
annotation('textbox',dim,'String',str,'FitBoxToText','on','Tag' , 'particleMovement');
pause(.001) %// pause 0.1 seconds to slow things down
drawnow  
frame = getframe(1);
im{iter} = frame2im(frame);
[A,map] = rgb2ind(im{iter},256);
if iter == 1
   imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.50);
else
   imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.50);
end

%-------------------------------------------------------------------------------------------------------------------------------------
    for particle = 1:swarmSize
        % Update velocity
        cognitive_component = c_1 * rand(1,nvars) .* (pbestPos(particle,:) - currentPos(particle,:));
        social_component = c_2 * rand(1,nvars) .* (gbestPos - currentPos(particle,:));
        velocityMatrix(particle,:) = inertia * velocityMatrix(particle,:) + cognitive_component + social_component;
        % Update position
        currentPos(particle,:) = currentPos(particle,:) + velocityMatrix(particle,:);
    end 
 
J_hist = [J_hist' [iter; costFunction(gbestPos'); mean(pBestJVec);mean(currentPosJVec)]]';
end

nn_params = gbestPos';
cost = costFunction(gbestPos');
trigger = 1;


end

