import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import math
from nn_training.matrix_dataset import MatrixDataset
from net import Net
from nn_training.early_stopping import EarlyStopping
from nn_training.evaluation import Evaluation
import sys
import getopt
import logging

DATA_PATH = './data/zscore/'
PLOT_PATH = './output/'
TRAINSET_PATH = DATA_PATH + 'train_set_partial_excluding_original.csv'
TESTSET_PATH = DATA_PATH + 'test_set_partial_excluding_original.csv'
TRAINSET_FREEZE_PATH = DATA_PATH + 'train_set_original_balance_part.csv'
TESTSET_FREEZE_PATH = DATA_PATH + 'test_set_original_part.csv'
ORIGINALSET_PATH = DATA_PATH + 'test_set_original_balanced.csv'

MODEL_PATH = './models/model_net.pt'

feature_collection = 'zscore'
batch_size = 64
epochs = 1
learning_rate = 1e-4
random_seed = 56
freeze_layers = None
freeze_epochs = 0
layers = 6
neurons = 16


class Train():
    def __init__(self, proceed=False):
        self.logger = logging.getLogger(__name__)
        self.epochs = epochs
        self.batch_size = batch_size

        if random_seed != None:
            self.logger.info("random_seed: %d", random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed + 12)
            np.random.seed(random_seed + 53)

        self.set_datasets(TRAINSET_PATH)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using {self.device} device")
        self.logger.info("device_count: %d", torch.cuda.device_count())
        if torch.cuda.is_available():
            self.logger.info("device: %s", torch.cuda.get_device_name())

        amount_features = MatrixDataset.get_amount_features_for_collection(feature_collection)
        self.model = Net(amount_features, layers, neurons).to(self.device)
        if (proceed):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # scheduler = scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
        self.criterion = nn.BCELoss()
        self.logger.info(self.model)


    def set_datasets(self, train_path, original_path=ORIGINALSET_PATH):
        df = pd.read_csv(train_path)
        train_df = df.sample(frac=0.8)
        val_df = df.drop(train_df.index)
        orig_df = pd.read_csv(original_path)
        self.logger.info("train len: %d", len(train_df))
        self.logger.info("val len: %d", len(val_df))
        self.logger.info("batch_size: %d", self.batch_size)
        self.logger.info(f'start learning rate: {learning_rate:.4e}')

        train_set = MatrixDataset(train_df, feature_collection)
        val_set = MatrixDataset(val_df, feature_collection)
        orig_set = MatrixDataset(orig_df, feature_collection)

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.orig_loader = torch.utils.data.DataLoader(orig_set, batch_size=self.batch_size, shuffle=True, pin_memory=True)


    def train(self):
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        original_loss = []
        original_accuracy = []

        self.early_stopping = EarlyStopping(patience=15, path=MODEL_PATH, delta=0.08)
        start = datetime.now()
        for epoch in range(self.epochs):
            running_train_loss, running_train_accuracy, f_one_train = self.train_one_epoch()
            train_loss.append(running_train_loss)
            train_accuracy.append(running_train_accuracy)

            running_val_loss, running_val_accuracy, f_one_validation = self.validation(self.val_loader)
            val_loss.append(running_val_loss)
            val_accuracy.append(running_val_accuracy)

            running_original_loss, running_original_accuracy , temp = self.validation(self.orig_loader)
            original_loss.append(running_original_loss)
            original_accuracy.append(running_original_accuracy)

            if True or epoch % math.ceil(self.epochs / 50) == 0:
                self.logger.info(f'[{epoch + 1}] train_loss: {running_train_loss:.10f}, val_loss: {running_val_loss:.10f}, '
                      f'train_accuracy: {running_train_accuracy:.2f}, val_accuracy: {running_val_accuracy:.2f}, '
                      f'train_f_one: {f_one_train:.2f}, val_f_one: {f_one_validation:.2f}, '
                    # f'learning rate: {scheduler.get_last_lr()[0]:.4e}'
                )

            # scheduler.step()
            self.early_stopping(running_val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

        end = datetime.now()
        self.logger.info(f'Finished Training in {(end - start).seconds // 60} minutes')
        self.create_plot(train_loss, train_accuracy, val_loss, val_accuracy)
        self.create_plot(train_loss, train_accuracy, val_loss, val_accuracy, original_loss, original_accuracy)
        return self.model


    def train_one_epoch(self):
        running_loss = 0.0
        true_positives, false_positives, true_negatives, false_negatives, length = 0, 0, 0, 0, 0
        self.model.train(True)
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs.to(self.device))
            # logging.debug('Outputs: ', ' '.join(f'{outputs[j]}' for j in range(len(outputs))))
            loss = self.criterion(outputs, labels.unsqueeze(1).double().to(self.device))
            loss.backward()
            self.optimizer.step()

            prediction = outputs.round().to(torch.int)
            truth = labels.unsqueeze(1).to(self.device)
            confusion_vector = prediction / truth
            true_positives += torch.sum(confusion_vector == 1).item()
            false_positives += torch.sum(confusion_vector == float('inf')).item()
            true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives += torch.sum(confusion_vector == 0).item()
            length += prediction.size(dim=0)

            running_loss += loss.item()
        accuracy = 100 * (true_positives + true_negatives) / length
        if true_positives != 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_one = 2 * precision * recall / (precision + recall)
        else:
            f_one = 0
        return running_loss / len(self.train_loader), accuracy, f_one


    def validation(self, loader):
        self.model.train(False)
        val_loss = 0.0
        true_positives, false_positives, true_negatives, false_negatives, length = 0, 0, 0, 0, 0

        for inputs, labels in loader:
            outputs = self.model(inputs.to(self.device))
            loss = self.criterion(outputs, labels.unsqueeze(1).double().to(self.device))
            val_loss += loss.item()

            prediction = outputs.round().to(torch.int)
            truth = labels.unsqueeze(1).to(self.device)
            confusion_vector = prediction / truth
            true_positives += torch.sum(confusion_vector == 1).item()
            false_positives += torch.sum(confusion_vector == float('inf')).item()
            true_negatives += torch.sum(torch.isnan(confusion_vector)).item()
            false_negatives += torch.sum(confusion_vector == 0).item()
            length += prediction.size(dim=0)

        accuracy = 100 * (true_positives + true_negatives) / length
        if true_positives != 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_one = 2 * precision * recall / (precision + recall)
        else:
            f_one = 0

        return val_loss / len(self.val_loader), accuracy, f_one


    def create_plot(self, train_loss, train_accuracy, val_loss, val_accuracy, orig_loss=None, orig_accuracy=None):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(train_loss, label='train')
        axs[0].plot(val_loss, label='valid')
        if (orig_loss):
            axs[0].plot(orig_loss, label='original')
        axs[0].legend()
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('loss')
        axs[0].set_yscale('log')

        axs[1].plot(train_accuracy, label='train')
        axs[1].plot(val_accuracy, label='valid')
        if (orig_accuracy):
            axs[1].plot(orig_accuracy, label='original')
        axs[1].legend()
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('accuracy')

        #fig.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        name = PLOT_PATH + f"loss_plot_{timestamp}.png"
        if (orig_loss):
            name = PLOT_PATH + f"loss_plot_with_original_{timestamp}.png"
        fig.savefig(name)
        plt.show()


def get_args():
    argv = sys.argv
    arg_help = "{0} -f <feature_collection> -l <layers> -n <neurons> -b <batch_size> -e <epochs>" \
               " -r <random_seed> -p (proceed)  --learning_rate <learning_rate> --no_evaluation (no evaluation)" \
               " -freeze_layers <number_of_freeze_layers> -freeze_epochs <number_of_freeze_epochs>".format(argv[0])
    opts, args = getopt.getopt(argv[1:], "hf:l:n:b:e:r:p", ["help", "feature_collection=", "layers=", "neurons=",
        "batch_size=", "epochs=", "random_seed=", "proceed", "learning_rate=", "no_evaluation",
        "freeze_layers=", "freeze_epochs="])
    proceed = False
    evaluation = True
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(2)
        elif opt in ("-f", "--feature_collection"):
            global feature_collection
            feature_collection = arg
        elif opt in ("-l", "--layers"):
            global layers
            layers = arg
        elif opt in ("-n", "--neurons"):
            global neurons
            neurons = arg
        elif opt in ("-b", "--batch_size"):
            global batch_size
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            global epochs
            epochs = int(arg)
        elif opt in ("-r", "--random_seed"):
            global random_seed
            random_seed = int(arg)
        elif opt in ("-p", "--proceed"):
            proceed = True
        elif opt in ("--learning_rate"):
            global learning_rate
            learning_rate = float(arg)
        elif opt in ("--no_evaluation"):
            evaluation = False
        elif opt in ("--freeze_layers"):
            global freeze_layers
            freeze_layers = int(arg)
        elif opt in ("--freeze_epochs"):
            global freeze_epochs
            freeze_epochs = int(arg)
    return proceed, evaluation

def _getWeightForHist(func):
    return func.weight.reshape(-1).detach().cpu().numpy()

def createWeightPlot(model):
    fig, axs = plt.subplots(1,  model.layers + 1, figsize=(5 * model.layers, 5))
    for i in range(model.layers - 1):
        axs[i].hist(_getWeightForHist(getattr(model, f'fc{i}')), label=f'fc{i}')
        axs[i].legend()
        axs[i].set_xlabel('weight')

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    name = PLOT_PATH + f"weights_{timestamp}.png"
    fig.savefig(name)
    plt.show()

def main():
    logging.info('Start main')
    proceed, evaluation = get_args()
    logging.info('set args')

    train = Train(proceed=proceed)
    model = train.train()
    createWeightPlot(model)

    if (freeze_layers != None and freeze_epochs > 0):
        for i in range(min(model.layers - 1, freeze_layers)):
            getattr(model, f'fc{i}').requires_grad_(False)
            getattr(model, f'dropout{i}').requires_grad_(False)
            getattr(model, f'activation{i}').requires_grad_(False)
        logging.info(f"Freeze {freeze_layers} layers")

        train.epochs = freeze_epochs
        train.batch_size = 16

        train.set_datasets(TRAINSET_FREEZE_PATH)
        model = train.train()
        createWeightPlot(model)

    if (evaluation):
        ev = Evaluation(feature_collection = feature_collection, layers = layers, neurons = neurons)
        logging.info('--------------------SYNTETIC DATASET--------------------')
        ev.eval(testset_path = TESTSET_PATH)
        logging.info('--------------------ORIGINAL DATASET PART (FOR FINE TUNING)--------------------')
        ev.eval(testset_path = TESTSET_FREEZE_PATH)
        logging.info('--------------------COMPLETE ORIGINAL DATASET--------------------')
        ev.eval(testset_path = ORIGINALSET_PATH)
        logging.info('----------' + feature_collection + '----------')


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    logging.basicConfig(filename=PLOT_PATH + f"log_{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    main()
