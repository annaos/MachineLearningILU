import pandas as pd
import data_files
from dataset.utils import Utils

DATA_PATH = '../data/'
DATASET_PATH = DATA_PATH + 'dataset_split_big_3.csv'
TRAINSET_PATH = DATA_PATH + 'train_set.csv'
TESTSET_PATH = DATA_PATH + 'test_set.csv'


df = pd.read_csv(DATASET_PATH)
df = Utils.generate_relative_features(df)
df = Utils.update_is_effective(df, factor=1.5)
df = Utils.cut_df_by_is_effective(df, n=2000 * 2)

normalized_df = Utils.normalize(df)
train_df = normalized_df.sample(frac=0.8)
test_df = normalized_df.drop(train_df.index)

print("train len: ", len(train_df))
print("test len: ", len(test_df))

train_df.to_csv(TRAINSET_PATH, index=False)
test_df.to_csv(TESTSET_PATH, index=False)

print('Done')

