import os
import sys
from compute_features import ComputeFeatures
from compute_features_random import ComputeFeaturesRandom
import pandas as pd
import data_files


def extend(dataset):
    if os.path.exists(data_files.DATASET_PATH):
        existed_df = pd.read_csv(data_files.DATASET_PATH)
        dataset = pd.concat([dataset, existed_df])


def main():
    print("got")
    args = sys.argv[1:]
    matrix_origin = args[0]

    label_df = pd.read_csv(data_files.MATRICES_PATH)
    print(f"got label df: {len(label_df)} entries")

    if matrix_origin == 'ss':
        compute_features = ComputeFeatures()
    elif matrix_origin == 'random':
        compute_features = ComputeFeaturesRandom()

    feature_df = compute_features.get_feature_df(label_df)
    if feature_df.size == 0:
        print("feature df is empty")
        exit()
    print("got feature df")

    dataset = compute_features.merge(label_df, feature_df)
    print("merged dfs")

    dataset.to_csv(data_files.DATASET_PATH, index=False)
    print("saved dataset")

if __name__ == "__main__":
    main()
