# Neural Network for Solving the Two-Spiral Problem
This is a simple implementation of a neural network trained using different optimization algorithms in order to solve the twin-spiral problem. The two-spiral problem is a particularly difficult problem that requires separating two logistic spirals from one another [1] [2]. 

![Two Spiral Problem](http://i.imgur.com/AB14SHC.png)

### Files:
1. **twin_spiral_vanilla.m** - the main file containing the neural network structure. From here, different functions are called. 
2. **nnCostFunction.m** - function that contains the feedforward and backpropagation methods for the neural network. Here, both the cost and the gradient is computed and returned back to the main function via a handling variable.
3. **randInitializeWeights.m** - function that randomly initializes the weights of the neural network in order for it to break symmetry.
4. **predict.m** - function that predicts the classes of the input data given the trained parameters for the neural network.
5. **visualizeData.m** - maps the generalization ability of the neural network given the trained parameters.

### Optimization Algorithms:
1. **particleSwarmOptimization.m** - takes the objective function, initial parameters, and hyperparameters and finds the optima using a global best PSO algorithm.
2. **differentialEvolution.m** - takes the objective function, initial parameters, and hyperparameters and finds the optima using the differential evolution algorithm.
3. **fmincg.m** - stands for Function minimize nonlinear conjugant gradient, an off-the-shelf algorithm that was used as a benchmark solution

### Generalization Ability
1. **Particle Swarm Optimization** (84.86%)  
![Generalization ability of PSO](http://i.imgur.com/JtMGhr8.png)  
2. **Differential Evolution** *in-progress*  
3. **Minimize Nonlinear Conjugant Gradient** (100.00%)  
![Generalization ability of FMINCG](http://i.imgur.com/SIGJKSa.png)

*References*  
[1] Kevin J. Lang and Michael J. Witbrock: Learning to Tell Two Spirals Apart. In: *Proceedings of the 1988 Connectionist Models Summer School*, Morgan Kauffman, 1998.  
[2] http://www.ibiblio.org/pub/academic/computer-science/neural-networks/programs/bench/two-spirals  

*(A more comprehensive explanation of the theory and the implementation results for the various optimization algorithms will be linked here as a separate documentation)*
