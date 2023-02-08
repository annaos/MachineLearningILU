import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, amount_features, layers = 6, neurons = 16):
        super().__init__()
        self.layers = layers
        # self.fc0 = nn.Linear(amount_features, 16)
        # self.fc1 = nn.Linear(16, 16)
        # self.fc2 = nn.Linear(16, 16)
        # self.fc3 = nn.Linear(16, 16)
        # self.fc4 = nn.Linear(16, 16)
        # self.fc5 = nn.Linear(16, 16)
        # self.fc6 = nn.Linear(16, 16)
        # self.fc7 = nn.Linear(16, 1)

        for i in range(self.layers):
            fc = f'fc{i}'
            if i == 0:
                setattr(self, fc, nn.Linear(amount_features, neurons))
            elif i == self.layers - 1:
                setattr(self, fc, nn.Linear(neurons, 1))
            else:
                setattr(self, fc, nn.Linear(neurons, neurons))

            dropout = f'dropout{i}'
            setattr(self, dropout, nn.Dropout(p=0.05))

            activation = f'activation{i}'
            setattr(self, activation, nn.LeakyReLU())

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        for i in range(self.layers - 1):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'activation{i}')(x)

        i += 1
        x = getattr(self, f'fc{i}')(x)
        x = self.sigmoid(x)
        return x
