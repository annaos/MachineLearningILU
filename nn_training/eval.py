import pandas as pd
from dataset import MatrixDataset
import torch
from net import Net

DATA_PATH = '../data/'
TESTSET_PATH = DATA_PATH + 'test_set.csv'
reduced_feature = 0
MODEL_PATH = "../models/model_net.pth"

def get_report():
    for features, labels in val_loader:
        features = features.to(device)

        predictions = torch.squeeze(model(features.to(device)).round().to(torch.int))
        error = 0
        prediction_effective = 0
        label_effective = 0
        error_prediction_effective = 0
        error_label_effective = 0
        for i, (pred, l) in enumerate(zip(predictions, labels)):
            if pred == 1:
                label_effective += 1
            if l == 1:
                prediction_effective += 1
            if pred != l:
                error += 1
                if l == 1:
                    error_label_effective += 1
                elif pred == 1:
                    error_prediction_effective += 1
        print('Test set: ', len(labels))
        print('Predicted effective: ', prediction_effective)
        print('Label effective: ', label_effective)
        print('Errors: ', error)
        print('Error, where predicted effective: ', error_prediction_effective)
        print('Error, where label effective: ', error_label_effective)

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
