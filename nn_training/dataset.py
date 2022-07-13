import numpy as np
import torch


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, feature_collection):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.feature_collection = feature_collection

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        features = self.data_df.iloc[index]
        isEffective = features.isEffective

        f_reduced = np.array([
            features.density, features.nonzeros, features.nsym, features.posdef, features.psym, features.rows
        ])

        f_full = np.array([
            features.rows, features.nonzeros, features.posdef, features.nsym, features.psym, features.density,
            features.avg_nnz, features.max_nnz, features.std_nnz,
            features.avg_row_block_count, features.std_row_block_count, features.min_row_block_count,
            features.max_row_block_count,
            features.avg_row_block_size, features.std_row_block_size, features.min_row_block_size,
            features.max_row_block_size,
            features.block_count
        ])

        f_reduced_1_without_sym = np.array([
            features.rows, features.nonzeros, features.nsym, features.density,
            features.avg_nnz, features.max_nnz, features.std_nnz,
            features.avg_row_block_count, features.std_row_block_count, features.min_row_block_count,
            features.max_row_block_count,
            features.avg_row_block_size, features.std_row_block_size, features.min_row_block_size,
            features.max_row_block_size,
            features.block_count
        ])

        f_reduced_2_relative = np.array([
            features.rows, features.nonzeros_relative, features.nsym, features.density,
            features.avg_nnz_relative, features.max_nnz_relative,
            features.std_nnz_relative,
            features.avg_row_block_count_relative, features.std_row_block_count_relative,
            features.min_row_block_count_relative, features.max_row_block_count_relative,
            features.avg_row_block_size_relative, features.std_row_block_size_relative,
            features.min_row_block_size_relative, features.max_row_block_size_relative,
            features.block_count_relative
        ])

        if self.feature_collection == 'full':
            return f_full, isEffective
        if self.feature_collection == 'not_sym':
            return f_reduced_1_without_sym, isEffective
        if self.feature_collection == 'relative':
            return f_reduced_2_relative, isEffective
        return f_reduced, isEffective

    def get_amount_features(self):
        if self.feature_collection == 'full':
            return 18
        if self.feature_collection == 'not_sym':
            return 16
        if self.feature_collection == 'relative':
            return 16
        return 6
