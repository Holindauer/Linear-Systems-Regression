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

        input = self.vectorize(A, b)    # <--- vectorize example components
        target = x.view(batch_size, -1) # <--- flatten x

        return input, target
    
    # Since the initial model will be an MLP, we need toconcat and vectorize the matrix A and vector b as inputs
    def vectorize(self, A: Tensor, b: Tensor) -> Tensor:

        batch_size, size, _ = A.shape  # Assuming A is of shape [batch_size, size, size]

        # Reshape A and b while keeping the batch dimension intact
        A_flat, b_flat = A.view(batch_size, -1), b.view(batch_size, -1)

        # Concatenate along the second dimension (columns)
        return torch.cat((A_flat, b_flat), dim=1)