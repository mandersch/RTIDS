from torch import nn
import torch 
from einops import rearrange

class RTIDS_Embedder(nn.Module):
    def __init__(self, dim, num_numerical_types=78):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases