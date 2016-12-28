# Neural Network for Solving the Two-Spiral Problem
This is a simple implementation of a 2-M-1 neural network trained using different optimization algorithms in order to solve the two-spiral problem. The two-spiral problem is a particularly difficult problem that requires separating two logistic spirals from one another [1] [2]. 

![Two Spiral Problem](http://i.imgur.com/AB14SHC.png)  

## Files Included:
### Neural Network: 
1. **twin_spiral_vanilla.m** - the main file containing the neural network structure. From here, different functions are called. 
2. **nnCostFunction.m** - function that contains the feedforward and backpropagation methods for the neural network. Here, both the cost and the gradient is computed and returned back to the main function via a handling variable.
3. **randInitializeWeights.m** - function that randomly initializes the weights of the neural network in order for it to break symmetry.
4. **predict.m** - function that predicts the classes of the input data given the trained parameters for the neural network.
5. **visualizeData.m** - maps the generalization ability of the neural network given the trained parameters.

### Optimization Algorithms:
1. **particleSwarmOptimization.m** - takes the objective function, initial parameters, and hyperparameters and finds the optima using a global best PSO algorithm.
2. **differentialEvolution.m** - takes the objective function, initial parameters, and hyperparameters and finds the optima using the differential evolution algorithm.
3. **fmincg.m** - stands for Function minimize nonlinear conjugant gradient, an off-the-shelf algorithm that was used as a benchmark solution

## Installation: 
First, make sure that you have MATLAB installed. Compatibility to open-source software, such as Octave, has not yet been tested. If you're set, then just clone this repository:

`$ git clone https://github.com/ljvmiranda921/two-spiral-neural-net.git`

## Usage:
### Parameter Setting:
Both PSO and DE have different parameters that one can experiment on. In this implementation, the parameters present are summarized in the table below

**Particle Swarm Optimization**   

| Parameter       | Description                                              |
|-----------------|----------------------------------------------------------|
| `maxIter`       | Number of iterations that the PSO algorithm will run     |
| `swarmSize`     | Number of particles in the search space.                 |
| `epsilon`       | Scattering degree of the particles during initialization |
| `c_1`           | Cognitive component (exploration parameter).             |
| `c_2`           | Social component (exploitation parameter).               |
| `inertiaWeight` | Inertia weight that controls the movement of particles.  |

**Differential Evolution**  

| Parameter        | Description                                              |
|------------------|----------------------------------------------------------|
| `genMax`         | Number of generations that the DE algorithm will run     |
| `population`     | Number of particles in the search space.                 |
| `epsilon_de`     | Scattering degree of the particles during initialization |
| `mutationF`      | Degree of mutation effect (exploration parameter)        |
| `recombinationC` | Degree of recombination effect (exploitation parameter)  |

## Implementation Notes:
### Generalization Ability
1. **Particle Swarm Optimization** (84.86%)  
<img  src="http://i.imgur.com/POaz0v1.png" alt="Cost Function" width="400px"> <img src="http://i.imgur.com/YCLmiCE.gif" alt="PSO" width="400px">
![Generalization ability of PSO](http://i.imgur.com/JtMGhr8.png)  
2. **Differential Evolution** *in-progress*  
3. **Minimize Nonlinear Conjugant Gradient** (100.00%)  
![Generalization ability of FMINCG](http://i.imgur.com/SIGJKSa.png)

## License 
This project is licensed under the MIT License - see the LICENSE.txt file for details

## References 
[1] Kevin J. Lang and Michael J. Witbrock: Learning to Tell Two Spirals Apart. In: *Proceedings of the 1988 Connectionist Models Summer School*, Morgan Kauffman, 1998.  
[2] http://www.ibiblio.org/pub/academic/computer-science/neural-networks/programs/bench/two-spirals  

*(A more comprehensive explanation of the theory and the implementation results for the various optimization algorithms will be linked here as a separate documentation)*
