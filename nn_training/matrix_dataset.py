import numpy as np
import torch
from dataset.utils import Utils


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, feature_collection='relative'):
        """
        Args:
            data_df (DataFrame): Dataframe with dataset.
            feature_collection (str): 'pure', 'relative', 'normalized', 'relative_normalized', 'all'
                            Default: 'relative'
        """
        self.data_df = data_df.replace([True], 1).replace([False], 0)
        self.feature_collection = feature_collection

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        features = self.data_df.iloc[index]
        is_effective = features.is_effective

        feature_list = Utils.get_features_list(self.feature_collection)
        f = np.array(features.get(feature_list)).astype('float64')
        return f, is_effective

    @staticmethod
    def get_amount_features_for_collection(feature_collection):
        feature_list = Utils.get_features_list(feature_collection)
        return len(feature_list)


    def get_amount_features(self):
        feature_list = Utils.get_features_list(self.feature_collection)
        return len(feature_list)
