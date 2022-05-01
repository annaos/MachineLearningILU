import torch.nn as nn
import torch.optim as optim
import torch
from matplotlib import pyplot as plt
import pandas as pd
from dataset import MatrixDataset
from datetime import datetime
from net import Net

DATA_PATH = '../data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'
TESTSET_PATH = DATA_PATH + 'test_set.csv'
LOSS_PLOT_PATH = DATA_PATH + 'loss_plot.png'

reduced_feature = 0
batch_size = 10
epochs = 20

df = pd.read_csv(DATASET_PATH).dropna()
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("test len: ", len(test_df))
test_df.to_csv(TESTSET_PATH, index=False)
train_set = MatrixDataset(train_df, reduced_feature)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print("device_count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device: ", torch.cuda.get_device_name())

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
    plt.savefig(LOSS_PLOT_PATH)
    return model

model = train(model, train_loader, epochs)
ts = int(datetime.timestamp(datetime.now()))
torch.save(model.state_dict(), f"../models/model_{epochs}_{len(train_df)}_{ts}.pt")

