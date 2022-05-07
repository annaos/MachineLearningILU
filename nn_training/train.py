import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch
from matplotlib import pyplot as plt
import pandas as pd
from dataset import MatrixDataset
from datetime import datetime
from net import Net
import time

DATA_PATH = '../data/'
TRAINSET_PATH = DATA_PATH + 'train_set.csv'
LOSS_PLOT_PATH = DATA_PATH + 'loss_plot.png'
TRAIN_LOSS_PLOT_PATH = DATA_PATH + 'train_loss_plot.png'
VAL_LOSS_PLOT_PATH = DATA_PATH + 'val_loss_plot.png'

reduced_feature = 0
batch_size = 32
epochs = 500
learning_rate = 100

df = pd.read_csv(TRAINSET_PATH).dropna()
train_df = df.sample(frac=0.9)
val_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("val len: ", len(val_df))
train_set = MatrixDataset(train_df, reduced_feature)
val_set = MatrixDataset(val_df, reduced_feature)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size//8, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print("device_count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device: ", torch.cuda.get_device_name())

model = Net(train_set.get_amount_features()).to(device)
model.double()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = scheduler.ExponentialLR(optimizer, gamma=0.9)
criterion = nn.BCELoss()

def train():
    train_loss = []
    train_accuracy = []
    val_loss = []

    start = time.time()
    for epoch in range(epochs):
    #    print('--------Epoch: ', epoch)
        running_train_loss, running_train_accuracy = train_one_epoch()
        train_accuracy.append(running_train_accuracy)
        train_loss.append(running_train_loss)

        running_val_loss = validation()
        val_loss.append(running_val_loss)
        scheduler.step()
        if epoch % (epochs/50) == 0:
            print(f'[{epoch + 1}] loss: {running_train_loss:.10f}, val_loss: {running_val_loss:.10f}, '
                  f'accuracy: {running_train_accuracy:.2f}, learning rate: {scheduler.get_last_lr()[0]:.4e}')
    end = time.time()
    print(f'Finished Training in {(end - start) // 60} minutes')

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(train_loss, label='train')
    axs[0].plot(val_loss, label='valid')
    axs[0].legend()
    axs[0].set_xlabel('epochs')
    axs[1].set_ylabel('loss')

    axs[1].plot(train_accuracy, label='accuracy')
    axs[1].legend()
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('percent')

    fig.tight_layout()
    plt.show()
    plt.savefig(LOSS_PLOT_PATH)
    return model

def train_one_epoch():
    running_loss = 0.0
    correct = 0
    model.train(True)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        # print('Outputs: ', ' '.join(f'{outputs[j]}' for j in range(len(outputs))))
        loss = criterion(outputs, labels.unsqueeze(1).double().to(device))
        loss.backward()
        optimizer.step()
        correct += (outputs.round().to(torch.int) == labels.unsqueeze(1).to(device)).sum().item()
        running_loss += loss.item()
    accuracy = 100 * correct / len(train_df)
    return running_loss / len(train_loader), accuracy


def validation():
    model.train(False)
    val_loss = 0
    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.unsqueeze(1).double().to(device))
        val_loss += loss.item()
    return val_loss / len(val_loader)


model = train()
ts = int(datetime.timestamp(datetime.now()))
torch.save(model.state_dict(), f"../models/model_{ts}_{epochs}_{len(train_df)}.pt")

