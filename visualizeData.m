%% ================================== Plot the original function ===================================
fprintf('Status: Displaying data\n');
cd('C:\Users\Lj Miranda\Documents\Waseda\2016-01 Fall\110400F Computational Intelligence\reports\report-ci-final\twin-spirals')
two_spiral = csvread('two_spiral.csv');
X = two_spiral(:,1:2);  % x-coordinate and y-coordinate
y = two_spiral(:,3);    % category [0/1]

%% ================================== Unroll the parameters and set-up test NN =====================
fprintf('\nLoading parameters from file.\n');
load('85_pso_theta1_finalrun.mat')
load('85_pso_theta2_finalrun.mat')
fprintf('Program paused. Press enter to continue.\n');
pause;

% set up the domain over which you want to visualize the decision
% boundary
xrange = [-0.5 0.5];
yrange = [-0.5 0.5];
dx = 0.001;

% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x_mesh, y_mesh] = meshgrid(xrange(1):dx:xrange(2), yrange(1):dx:yrange(2));

% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
image_size = size(x_mesh);

% make (x,y) pairs as a bunch of row vectors.
xy = [x_mesh(:) y_mesh(:)]; 

% perform prediction.
pred = predict(Theta1, Theta2, xy);
my_predictions = [xy, pred];

% reshape the pred (which contains the class label) into an image.
decisionmap = reshape(pred, image_size);

figure;


imagesc(xrange,yrange,decisionmap);
hold on;
gscatter(X(:,1),X(:,2),y,'rb','xo')
set(gca, 'ydir','normal');

cmap = [0.60 0.60 1.0; 1.0 1.0 0.60];
colormap(cmap);

xlabel('x');
ylabel('y');

