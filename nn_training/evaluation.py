import pandas as pd
from nn_training.matrix_dataset import MatrixDataset
import torch
from nn_training.net import Net
import numpy as np
import logging

class Evaluation():
    def __init__(self, feature_collection, layers, neurons, model_path = './models/model_net.pt'):
        self.logger = logging.getLogger(__name__)
        self.feature_collection = feature_collection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        amount_features = MatrixDataset.get_amount_features_for_collection(feature_collection)
        self.model = Net(amount_features, layers, neurons).to(self.device).double()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


    def __get_report(self, val_loader):
        for features, truth in val_loader:
            prediction = torch.squeeze(self.model(features.to(self.device)).round().to(torch.int))
            self.print_stats(truth, prediction)


    def print_stats(self, truth_labels, predictions):
        prediction_effective = predictions.sum().item()
        label_effective = truth_labels.unsqueeze(1).to(self.device).sum().item()

        confusion_vector = predictions / truth_labels.to(self.device)
        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        accuracy = 100 * (true_positives + true_negatives) / len(truth_labels)
        if true_positives != 0:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_one = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f_one = 0, 0, 0

        confusion_matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]
        self.logger.info('Test set: %d', len(truth_labels))
        self.logger.info('Label effective: %d', label_effective)
        self.logger.info('Predicted effective: %d', prediction_effective)
        self.logger.info('Errors: %d', false_negatives + false_positives)
        self.logger.info(f'Confusion matrix: $\\begin{{array}}{{rrr}} {true_positives} & {false_negatives} \\\\ {false_positives} & {true_negatives} \\\\ \\end{{array}}$')
        self.logger.info(np.matrix(confusion_matrix))

        self.logger.info('Error, where predicted effective (false positive): %d', false_positives)
        self.logger.info('Error, where label effective (false negative): %d', false_negatives)
        self.logger.info('Accuracy: %f', accuracy)
        self.logger.info('Precision: %f', precision)
        self.logger.info('Recall: %f', recall)
        self.logger.info('F_one: %f', f_one)


    def __print_results(self, val_loader):
        for i, data in enumerate(val_loader, 0):
            features, labels = data
            self.logger.info('GroundTruth: %s', ' '.join(f'{int(labels[j])}' for j in range(len(labels))))

            predicted = torch.squeeze(self.model(features.to(self.device)).round().to(torch.int))
            self.logger.info('Predicted  : %s', ' '.join(f'{predicted[j]}' for j in range(len(predicted))))


    def eval(self, testset_path = './data/test_set.csv', print_results = False):
        test_df = pd.read_csv(testset_path)
        test_set = MatrixDataset(test_df, self.feature_collection)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, pin_memory=True)

        if print_results:
            self.__print_results(val_loader)
        self.__get_report(val_loader)
