import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import math
from dataset import MatrixDataset
from net import Net
from early_stopping import EarlyStopping

DATA_PATH = '../data/'
TRAINSET_PATH = DATA_PATH + 'train_set.csv'
LOSS_PLOT_PATH = DATA_PATH + 'loss_plot.png'
MODEL_PATH = '../models/model_net.pt'

feature_collection = 'relative'
batch_size = 4
epochs = 10000
learning_rate = 1e-4
random_seed = 37

if random_seed != None:
    print("random_seed: ", random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed + 12)
    np.random.seed(random_seed + 53)
early_stopping = EarlyStopping(patience=20, path=MODEL_PATH, delta=0.1)

df = pd.read_csv(TRAINSET_PATH).dropna()
train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("val len: ", len(val_df))
print("batch_size: ", batch_size)
print(f'start learning rate: {learning_rate:.4e}')

train_set = MatrixDataset(train_df, feature_collection)
val_set = MatrixDataset(val_df, feature_collection)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print("device_count: ", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device: ", torch.cuda.get_device_name())

model = Net(train_set.get_amount_features()).to(device)
model.double()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
criterion = nn.BCELoss()
print(model)


def train():
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []

    start = datetime.now()
    for epoch in range(epochs):
        running_train_loss, running_train_accuracy, f_one_train = train_one_epoch()
        train_accuracy.append(running_train_accuracy)
        train_loss.append(running_train_loss)

        running_val_loss, running_val_accuracy, f_one_validation = validation()
        val_accuracy.append(running_val_accuracy)
        val_loss.append(running_val_loss)

        if True or epoch % math.ceil(epochs / 50) == 0:
            print(f'[{epoch + 1}] train_loss: {running_train_loss:.10f}, val_loss: {running_val_loss:.10f}, '
                  f'train_accuracy: {running_train_accuracy:.2f}, val_accuracy: {running_val_accuracy:.2f}, '
                  f'train_f_one: {f_one_train:.2f}, val_f_one: {f_one_validation:.2f}, '
                # f'learning rate: {scheduler.get_last_lr()[0]:.4e}'
            )

        # scheduler.step()
        early_stopping(running_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    end = datetime.now()
    print(f'Finished Training in {(end - start).seconds // 60} minutes')
    create_plot(train_loss, val_loss, train_accuracy, val_accuracy)
    return model


def train_one_epoch():
    running_loss = 0.0
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
    model.train(True)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        # print('Outputs: ', ' '.join(f'{outputs[j]}' for j in range(len(outputs))))
        loss = criterion(outputs, labels.unsqueeze(1).double().to(device))
        loss.backward()
        optimizer.step()

        prediction = outputs.round().to(torch.int)
        truth = labels.unsqueeze(1).to(device)
        confusion_vector = prediction / truth
        true_positives += torch.sum(confusion_vector == 1).item()
        false_positives += torch.sum(confusion_vector == float('inf')).item()
        true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives += torch.sum(confusion_vector == 0).item()

        running_loss += loss.item()
    accuracy = 100 * (true_positives + true_negatives) / len(train_df)
    if true_positives != 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f_one = 2 * precision * recall / (precision + recall)
    else:
        f_one = 0
    return running_loss / len(train_loader), accuracy, f_one


def validation():
    model.train(False)
    val_loss = 0.0
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0

    for inputs, labels in val_loader:
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.unsqueeze(1).double().to(device))
        val_loss += loss.item()

        prediction = outputs.round().to(torch.int)
        truth = labels.unsqueeze(1).to(device)
        confusion_vector = prediction / truth
        true_positives += torch.sum(confusion_vector == 1).item()
        false_positives += torch.sum(confusion_vector == float('inf')).item()
        true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives += torch.sum(confusion_vector == 0).item()

    accuracy = 100 * (true_positives + true_negatives) / len(val_df)
    if true_positives != 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f_one = 2 * precision * recall / (precision + recall)
    else:
        f_one = 0

    return val_loss / len(val_loader), accuracy, f_one


def create_plot(train_loss, val_loss, train_accuracy, val_accuracy):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(train_loss, label='train')
    axs[0].plot(val_loss, label='valid')
    axs[0].legend()
    axs[0].set_xlabel('epochs')
    axs[0].set_ylabel('loss')

    axs[1].plot(train_accuracy, label='train')
    axs[1].plot(val_accuracy, label='valid')
    axs[1].legend()
    axs[1].set_xlabel('epochs')
    axs[1].set_ylabel('accuracy')

    fig.tight_layout()
    plt.show()
    plt.savefig(LOSS_PLOT_PATH)


model = train()

import eval
