import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, amount_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(amount_features)
        self.fc1 = nn.Linear(amount_features, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sm(x)
        return x
