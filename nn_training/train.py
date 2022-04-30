import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
from dataset import MatrixDataset
import data_files

reduced_feature = 0
batch_size = 10

df = pd.read_csv(data_files.DATASET_PATH).dropna()
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("test len: ", len(test_df))
test_df.to_csv("test_set.csv", index=False)
train_set = MatrixDataset(train_df, reduced_feature)
test_set = MatrixDataset(test_df, reduced_feature)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print("device_count: ", torch.cuda.device_count())
print("device: ", torch.cuda.get_device_name())

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

model = Net(train_set.get_amount_features()).to(device)

model.double()
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.BCELoss()

def train(model, train_loader, num_epoch):
    loss_values = []

    for epoch in range(num_epoch):  # loop over the dataset multiple times
    #    print('--------Epoch: ', epoch)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            labels = labels.unsqueeze(1)
            labels = labels.double()

            optimizer.zero_grad()

            outputs = model(inputs.to(device))
            #print('Outputs: ', ' '.join(f'{outputs[j]}' for j in range(len(outputs))))

            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % 100 == 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.10f}')
        loss_values.append(running_loss)

    print('Finished Training')

    plt.plot(loss_values)
    plt.show()
    return model

def test(matrix_data, labels):
    print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(len(labels))))

    outputs = model(matrix_data.to(device))
    predicted = torch.squeeze(outputs.round().to(torch.int))
    print('Predicted: ', ' '.join(f'{predicted[j]}' for j in range(len(predicted))))


model = train(model, train_loader, 200)
torch.save(model.state_dict(), data_files.MODEL_PATH)

dataiter = iter(val_loader)
matrix_data, labels = dataiter.next()
#for i, data in enumerate(val_loader, 0):
#    matrix_data, labels = data
test(matrix_data, labels)

