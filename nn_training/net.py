import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, amount_features):
        super().__init__()
        self.fc0 = nn.Linear(amount_features, 16)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.05)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        input = x
        x = self.fc0(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.fc4(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
