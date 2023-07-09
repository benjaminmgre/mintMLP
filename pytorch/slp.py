import torch
from torch import nn


class SingleLayerPerceptron(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.layer_2 = nn.Linear(in_features=n_hidden, out_features=n_output)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        return x
