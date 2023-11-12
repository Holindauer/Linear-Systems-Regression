# Linear-Systems-Regression
Estimate Ax=b Solution with Neural Net


Project Plan and Inspo:
----------------------------------------------------------------------------------------------
This project began after a friend of mine told me about an iterative method for solving large sparse systems of linear equations called conjugate gradient. He was telling me that one of the applications of this method is for speeding up the training of neural networks. This sparked a thought that it seemed perhaps an even more efficient way to estimate sparse systems of linear equations would be to just train a neural net to do it. Neural networks are just funciton aproximators at the end of the day. This project is an attempt to find that function approximation.



Project Description:
----------------------------------------------------------------------------------------------
The code for this project is structured in a simple way. There are model.py files, train.py files, and generate_dataset.py. If you are familiar with machine learning and pytorch, the only part of this that needs explanation is the dataset: 
  - During the training loop, the class Dataset class within generate_dataset.py is called to generate a matrix A and vector x. These can be either sparse or dense, although sparse matricies are the intention of the project. A and x are m
  mutliplied to create a vector b. A and b are used as input features, and x is used as the target output. As such, there is not a static/traditional dataset for this project, but rather one that is created on the fly. 

There are also satelite training files like early_stopping.py to regulate the training process.
