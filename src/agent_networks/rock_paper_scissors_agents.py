"""
Simple agents to play the toy game: rock-paper-scissors.
"""

import torch
from torch.nn import Module
from torch.nn import Parameter


class TabularRPSPlayer(Module):
    """
    This architecture is too simple, it does not learn anything!
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.super_complex_neural_network = Parameter(
            torch.randn((3), device=device)
        )
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, input):
        return self.softmax(self.super_complex_neural_network * input)


class MLPRPSPlayer(Module):
    """
    Basic multi-layer perceptron agent network architecture to learn RPS.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRPSPlayer, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

