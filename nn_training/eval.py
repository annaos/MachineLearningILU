import pandas as pd
from dataset import MatrixDataset
import torch
from net import Net

DATA_PATH = '../data/'
TESTSET_PATH = DATA_PATH + 'test_set.csv'
reduced_feature = 0
MODEL_PATH = "../models/model_net.pt"

def get_report():
    for features, labels in val_loader:
        features = features.to(device)

        predictions = torch.squeeze(model(features.to(device)).round().to(torch.int))
        prediction_effective = 0
        label_effective = 0
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i, (pred, l) in enumerate(zip(predictions, labels)):
            if l == 1:
                label_effective += 1
                if pred == 1:
                    prediction_effective += 1
                    true_pos += 1
                if pred == 0:
                    false_neg += 1
            if l == 0:
                if pred == 1:
                    prediction_effective += 1
                    false_pos += 1
                if pred == 0:
                    true_neg += 1

        accuracy = 100 * (true_pos + true_neg)/ len(labels)
        if true_pos != 0:
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f_one = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f_one = 0, 0, 0

        print('Test set: ', len(labels))
        print('Label effective: ', label_effective)
        print('Predicted effective: ', prediction_effective)
        print('Errors: ', false_neg + false_pos)
        print('Error, where predicted effective (false positive): ', false_pos)
        print('Error, where label effective (false negative): ', false_neg)
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F_one: ', f_one)

def get_results():
    for i, data in enumerate(val_loader, 0):
        features, labels = data
        print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(len(labels))))

        predicted = torch.squeeze(model(features.to(device)).round().to(torch.int))
        print('Predicted  : ', ' '.join(f'{predicted[j]}' for j in range(len(predicted))))

test_df = pd.read_csv(TESTSET_PATH)
test_set = MatrixDataset(test_df, reduced_feature)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                         shuffle=False, pin_memory=True)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = Net(test_set.get_amount_features()).to(device)
model.double()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

get_results()
get_report()
