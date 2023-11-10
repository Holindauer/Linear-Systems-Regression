import numpy 
import torch
from torch import Tensor    

"""
    The Dataset Class is used to generate a batch of square matricies A, and vector x. Each matrix  and vector 
    are generated using He Initialization and are then normalized using Batch Normalization. For each pair of A 
    and x, their product vector b is computed to create a batch of Ax = b. The batch is returned as a Torch Tensor.
"""
class Dataset:

    def __init__(self):
        pass

    def gen_batch(self, batch_size :int, size :int) -> Tensor:
        A = torch.randn(batch_size, size, size)
        x = torch.randn(batch_size, size, 1)        
        b = torch.bmm(A, x) # <--- bmm ==  batch matrix multiplication

        input = self.vectorize(A, b) # <--- vectorize example components

        return input, x
    
    # Since the initial model will be an MLP, we need toconcat and vectorize the matrix A and vector b as inputs
    def vectorize(self, A :Tensor, b :Tensor) -> Tensor:
        return torch.cat((A.view(-1), b.view(-1)))
