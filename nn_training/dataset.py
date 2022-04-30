import numpy as np
import torch
from torchvision.transforms import transforms


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, reduced_degree):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
        self.reduced_degree = reduced_degree

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        features = sample[10:]
        isEffective = sample.isEffective

        reduced = np.array([features.density,  features.nonzeros, features.nsym, features.posdef, features.psym, features.rows])
        full = np.array([features.rows, features.nonzeros, features.posdef, features.nsym, features.psym, features.density,
                         features.avg_nnz, features.max_nnz, features.std_nnz,
                         features.avg_row_block_count, features.std_row_block_count, features.min_row_block_count, features.max_row_block_count,
                         features.avg_row_block_size, features.std_row_block_size, features.min_row_block_size, features.max_row_block_size,
                         features.block_count])
        if self.reduced_degree == 3:
            return reduced, isEffective
        return full, isEffective

    def get_amount_features(self):
        if self.reduced_degree == 3:
            return 6
        if self.reduced_degree == 0:
            return 18
        return 18
