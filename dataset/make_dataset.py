import os
from compute_features import get_feature_df
import pandas as pd
import data_files

label_df = pd.read_csv(data_files.MATRICES_PATH)

print("got label df")
feature_df = get_feature_df(label_df)
if feature_df.size == 0:
    print("feature df is empty")
    exit()
print("got feature df")

dataset = label_df.merge(feature_df, left_on="ProblemId", right_on="id")
if os.path.exists(data_files.DATASET_PATH):
    existed_df = pd.read_csv(data_files.DATASET_PATH)
    dataset = pd.concat([dataset, existed_df])
print("merged dfs")

dataset.to_csv(data_files.DATASET_PATH, index=False)
print("saved dataset")

df = pd.read_csv(data_files.DATASET_PATH).dropna()
train_df = df.sample(frac=0.8)
val_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("val len: ", len(val_df))
val_df.to_csv(data_files.TESTSET_PATH, index=False)
