import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, A_size :int) -> None:
        super(Model, self).__init__()
        self.A_size = A_size
        self.input_features = (A_size ** 2) + A_size # A and b elements

        self.fc1 = nn.Linear(self.input_features, 100)
        self.fc2 = nn.Linear(100, A_size) # <--- output is a vector x of size A_size

    def forward(self, x :torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
