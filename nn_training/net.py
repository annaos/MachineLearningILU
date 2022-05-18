import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, amount_features):
        super().__init__()
        self.fc0 = nn.Linear(amount_features, 32)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 1)

        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc0(x)
        x = self.activation(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
