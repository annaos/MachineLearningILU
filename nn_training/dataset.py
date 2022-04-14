import numpy as np
import torch
from torchvision.transforms import transforms


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df.dropna().replace([True], 1).replace([False], 0)
      #  self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        features = sample[10:]
        isEffective = sample.isEffective

        reduced = np.array([features.density,  features.nonzeros, features.nsym, features.posdef, features.psym, features.rows])
      #  return self.transform(reduced), self.transform(isEffective)
      #  return self.transform(reduced), self.transform(isEffective)
        return reduced, isEffective
