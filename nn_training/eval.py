import pandas as pd
from dataset import MatrixDataset
import torch
from net import Net
import numpy as np

DATA_PATH = '../data/'
TESTSET_PATH = DATA_PATH + 'test_set.csv'
feature_collection = 'relative'
MODEL_PATH = "../models/model_net.pt"


def get_report():
    for features, truth in val_loader:
        prediction = torch.squeeze(model(features.to(device)).round().to(torch.int))

        prediction_effective = prediction.sum().item()
        label_effective = truth.unsqueeze(1).to(device).sum().item()

        confusion_vector = prediction / truth.to(device)
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        accuracy = 100 * (true_positives + true_negatives) / len(truth)
        if true_positives != 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_one = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f_one = 0, 0, 0

        confusion_matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]
        print('Test set: ', len(truth))
        print('Label effective: ', label_effective)
        print('Predicted effective: ', prediction_effective)
        print('Errors: ', false_negatives + false_positives)
        print('Confusion matrix:')
        print(np.matrix(confusion_matrix))

        print('Error, where predicted effective (false positive): ', false_positives)
        print('Error, where label effective (false negative): ', false_negatives)
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F_one: ', f_one)


def get_results():
    for i, data in enumerate(val_loader, 0):
        features, labels = data
        print('GroundTruth: ', ' '.join(f'{int(labels[j])}' for j in range(len(labels))))

        predicted = torch.squeeze(model(features.to(device)).round().to(torch.int))
        print('Predicted  : ', ' '.join(f'{predicted[j]}' for j in range(len(predicted))))


test_df = pd.read_csv(TESTSET_PATH)
test_set = MatrixDataset(test_df, feature_collection)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net(test_set.get_amount_features()).to(device)
model.double()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# get_results()
get_report()
