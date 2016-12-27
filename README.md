# Neural Network for Solving the Twin-Spiral Problem
This is a simple implementation of a neural network trained using different optimization algorithms in order to solve the twin-spiral problem. The twin-spiral (or two-spiral) problem is a particularly difficult problem that requires separating two logistic spirals from one another [1] [2]. 

### Files:
1. **twin_spiral_vanilla.m** - the main file containing the neural network structure. From here, different functions are called. 
2. **nnCostFunction.m** - contains the feedforward and backpropagation methods for the neural network. Here, both the cost and the gradient is computed and returned back to the main function via a handling variable.
3. **randInitializeWeights.m** - randomly initializes the weights of the neural network in order for it to break symmetry.
4. 

*References*  
[1] Kevin J. Lang and Michael J. Witbrock: Learning to Tell Two Spirals Apart. In: *Proceedings of the 1988 Connectionist Models Summer School*, Morgan Kauffman, 1998.  
[2] http://www.ibiblio.org/pub/academic/computer-science/neural-networks/programs/bench/two-spirals
