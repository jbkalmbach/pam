import torch
from torch import nn

_all_ = ['defaultDiscriminator', 'defaultGenerator']


class defaultDiscriminator(nn.Module):

    def __init__(self, X_dim, h_dim, dropout_frac=0.2):
        super(defaultDiscriminator, self).__init__()

        self.d = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 2*h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_frac),
            torch.nn.Linear(2*h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 1),
            torch.nn.Sigmoid()
            )

    def forward(self, x, c=None):

        y = self.d(x)
        return y


class defaultGenerator(nn.Module):

    def __init__(self, X_dim, h_dim, dropout_frac=0.2):
        super(defaultGenerator, self).__init__()

        self.g = torch.nn.Sequential(
            torch.nn.Linear(X_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, 2*h_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_frac),
            torch.nn.Linear(2*h_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, X_dim)
            )

    def forward(self, x, c=None):

        y = self.g(x)
        return y
