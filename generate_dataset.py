import torch
from torch import Tensor    

"""
    The Dataset Class is used to generate a batch of square matricies A, and vector x. Each matrix  and vector 
    are generated using He Initialization and are then normalized using Batch Normalization. For each pair of A 
    and x, their product vector b is computed to create a batch of Ax = b. The batch is returned as a Torch Tensor.
"""
class Dataset:

    def __init__(self, matrix_type :str, fill_percentage :float = 0) -> None:
        
        # matrix type refers to whether the matrix should be dense or sparse
        self.matrix_type = matrix_type

        #  fill percentage is the percentage of sparse matrix elements that should be filled
        self.fill_percentage = fill_percentage

    def gen_batch(self, batch_size :int, size :int) -> Tensor:
        
        # create batch based on matrix type
        if self.matrix_type == "dense":
            A, b, x = self.gen_dense(batch_size, size) 
        elif self.matrix_type == "sparse":
            A, b, x = self.gen_sparse(batch_size, size)

        input = self.vectorize(A, b)    # <--- vectorize example components
        target = x.view(batch_size, -1) # <--- flatten x

        return input, target
    

    """gen_dense is used to generate a batch of dense matricies A, vector x, and their products vector b."""
    def gen_dense(self, batch_size :int, size :int) -> Tensor:

        A = torch.randn(batch_size, size, size)
        x = torch.randn(batch_size, size, 1)        
        b = torch.bmm(A, x) # <--- bmm ==  batch matrix multiplication

        return A, b, x
    
    """gen_sparse is used to generate a batch of sparse matricies A, vector x, and their products vector b.
    The sparse matricies are generated """
    def gen_sparse(self, batch_size: int, size: int) -> Tensor:
        batch_A = []
        batch_x = []
        batch_b = []
        
        # compute num elements and num non-zero elements in A
        num_elements = size * size
        num_nonzero = int(num_elements * self.fill_percentage)

        for _ in range(batch_size):
            # Random indices for non-zero elements in A
            indices = torch.randperm(num_elements)[:num_nonzero]     # <--- randperm returns a random permutation of integers from 0 to num_elements
            indices = torch.stack((indices // size, indices % size)) # <--- stack indices along the first dimension (rows)

            # Values for A from a normal distribution (He initialization)
            values = torch.randn(num_nonzero) * torch.sqrt(torch.tensor(2. / size))

            # Create sparse tensor for A 
            sparse_A = torch.sparse_coo_tensor(indices, values, (size, size)) # <--- only non-zero elements are stored

            # Dense vector x from a normal distribution (He initialization)
            x = torch.randn(size, 1) * torch.sqrt(torch.tensor(2. / size))

            # Compute b using dense matrix multiplication
            b = torch.sparse.mm(sparse_A, x)

            batch_A.append(sparse_A.to_dense()) # <--- convert sparse_A to dense, ie fill in the zeros
            batch_x.append(x)
            batch_b.append(b)

        # Stack tensors to create batch
        return torch.stack(batch_A), torch.stack(batch_b), torch.stack(batch_x)


       
    # Since the initial model will be an MLP, we need toconcat and vectorize the matrix A and vector b as inputs
    def vectorize(self, A: Tensor, b: Tensor) -> Tensor:

        batch_size, size, _ = A.shape  # Assuming A is of shape [batch_size, size, size]

        # Reshape A and b while keeping the batch dimension intact
        A_flat, b_flat = A.view(batch_size, -1), b.view(batch_size, -1)

        # Concatenate along the second dimension (columns)
        return torch.cat((A_flat, b_flat), dim=1)