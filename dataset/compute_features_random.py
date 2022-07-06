import ssgetpy
from scipy.io import mmread
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import more_itertools as mit
from collections import defaultdict
import data_files
import os

from dataset.compute_features import ComputeFeatures


class ComputeFeaturesRandom(ComputeFeatures):

    def read_matrix(self, counter):
        mtx = mmread('../../matlab/random_matrices/' + str(counter) + '.m')
        return mtx

    def get_feature_df(self, label_df):
        feature_dict = dict()

        for i in range(len(label_df)):
            counter = label_df.counter[i]
            if i % 100 == 0:
                print(f'reading matrix {counter}')
            mtx = self.read_matrix(counter)

            feature_dict[counter] = dict()
            feature_dict[counter]["counter"] = counter
            feature_dict[counter]["rows"] = mtx.shape[0]
            feature_dict[counter]["cols"] = mtx.shape[1]
            feature_dict[counter]["nonzeros"] = mtx.count_nonzero()
            feature_dict[counter]["posdef"] = False
            feature_dict[counter]["psym"], feature_dict[counter]["nsym"] = self.p_n_symmetry(mtx)

            feature_dict[counter].update(self.compute_common_features(mtx))

        return pd.DataFrame(data=feature_dict).T

    def merge(self, label_df, feature_df):
        return label_df.merge(feature_df, on='counter')