import torch
import torch.nn as nn


# Create network for backpropogation optimization
class BPNet(nn.Module):
    def __init__(self, dims, device, num_classes=10):
        super().__init__()

        self.layers = []
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.device = device

        for d in range(len(dims) - 1):
            self.layers += [
                nn.Linear(dims[d], dims[d + 1]),
                nn.LayerNorm(dims[d + 1]),
                self.relu,
            ]
        self.layers += [nn.Linear(dims[-1], num_classes)]

        self.f = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.softmax(self.f(x))
