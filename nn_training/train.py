import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
from dataset import MatrixDataset

DATA_PATH = '../data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'
MODEL_PATH = DATA_PATH + 'model_net.pth'

df = pd.read_csv(DATASET_PATH).dropna()
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("test len: ", len(test_df))
test_df.to_csv("test_set.csv", index=False)
train_set = MatrixDataset(train_df)
test_set = MatrixDataset(train_df) #TODO ONLY FOR DEBUG
batch_size = 2

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, pin_memory=True)
# device = torch.device("cuda:2")
# print("device_count: ", torch.cuda.device_count())
# print("device: ", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model = nn.Sequential(
#     nn.BatchNorm1d(6),
#
#     nn.Linear(6, 447),
#     nn.ReLU(),
# #    nn.BatchNorm1d(?),
#
#     nn.Linear(447, 1324),
#     nn.ReLU(),
# #    nn.BatchNorm1d(?),
#
#     nn.Linear(1324, 2),
#     nn.Softmax(dim=1)
# ).to(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(6)
        self.fc1 = nn.Linear(6, 447)
        self.fc2 = nn.Linear(447, 1324)
        self.fc3 = nn.Linear(1324, 2)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
   #     x = self.sm(x)
        return x

model = Net()

model.double()
optimizer = optim.Adam(model.parameters(), lr=0.01386)
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
#criterion = nn.NLLLoss()

def train(model, train_loader, num_epoch):
    loss_values = []

    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss_values.append(running_loss)
       # loss_values.append(running_loss / len(train_set))

    print('Finished Training')

    plt.plot(loss_values)
    plt.show()
    return model

def test(model, val_loader):
    dataiter = iter(val_loader)
    matrix_data, labels = dataiter.next()
    print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(len(labels))))

    outputs = model(matrix_data)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(len(predicted))))


model = train(model, train_loader, 500)
torch.save(model.state_dict(), MODEL_PATH)

test(model, val_loader)
