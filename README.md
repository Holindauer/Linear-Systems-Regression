# Linear-Systems-Regression
Estimate Ax=b Solution with Neural Net


Project Plan:
----------------------------------------------------------------------------------------------
I had an idea this morning to train a neural net that can estimate the solution to a system of 
linear equations. I know of some iterative linear systems solvers like conjugate gradient for 
estimating x in Ax=b that have applications in efficient neural net optimization. I wonder whether 
a neural net would be more efficient at this task than an iterative process. That is the motivation
and curiosity behind this project. 

The plan is to implement a neural network solution to the problem of finding x in Ax=b. This suggests 
a few design considerations off the bat. 

  - The first would be that the A in this case, if this model is to have some application in nn optimization
     is to have A be a dense matrix of gaussian noise (since net params are randomly initialized).
    
  - The second would be to use a loss function like MSE to determine the performance of the estimation

  - The third would be that the dataset could be created within the training loop as need. Meaning that
    within an epoch, a batch of matricies A and vectors x could be generated randomly, their products (b)
    computed. The batch of A and b would then go on to be features and x would be the target. 

    While this would add to the time and compute required to train the net, it would save the training
    script from having to lug around a gigantic memory intensive dataset of dense matricies.

    It would also sacrifice reproducibility.



  
