# Linear-Systems-Regression
Estimate Ax=b Solution with Neural Net

How to Use:
----------------------------------------------------------------------------------------------
Running the following command will train a neural network with on the task of approximating x in Ax=b using the given input parameters. Using the A_size, matrix_type, and fill_percentage parameters, a random dataset of matricies Ax=b will be constructed at each epoch to train the model. The final model will be saved as a .pt file.

  python3 main.py --patience [PATIENCE] --min_delta [MIN_DELTA] --epochs [EPOCHS] --val_num_batches [VAL_NUM_BATCHES] --batch_size [BATCH_SIZE] --lr [LEARNING_RATE] --A_size [A_SIZE] --matrix_type [MATRIX_TYPE] --fill_percentage [FILL_PERCENTAGE]

- A_size refers to an integer argument of the number of unknowns the net is solving for in the system.
- matrix_type can either be "dense" or "sparse".
- fill_percentage refers to the percentage of non zero values in a sparse matrix.
- patience and min_delta are early stopping parameters reffering to many consecutive epochs without improvement will be allowed before stopping training and the minimum decrease in loww considered an improvement respectively.
- As well as some standard training hyperparameters: epchs, lr, batch_size, and val_num_batches (number of batched computed during validation).
  
Project Inspo:
----------------------------------------------------------------------------------------------
I had learned about some iterative linear systems of equation solvers like conjugate gradient a while back it sparked an interest in how well a Neural Network would do at the task. NNs are function approximators so it seemed totally possible. 

