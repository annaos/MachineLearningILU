import os
import sys
from compute_features import ComputeFeatures
from compute_features_random import ComputeFeaturesRandom
import pandas as pd
import data_files
from utils import Utils


def main():
    args = sys.argv[1:]
    matrix_origin_type = args[0]

    df = pd.read_csv(data_files.MATRICES_PATH)

    not_conv_df = df[df.conv0.eq(0) & df.conv1.eq(0)]
    label_df = df.drop(not_conv_df.index)
    label_df = Utils.update_is_effective(label_df, factor=1.5)
    label_df = Utils.cut_df_by_is_effective(label_df, n=2000 * 2)

    print(f"got label df: {len(label_df)} entries")

    if matrix_origin_type == 'ss':
        compute_features = ComputeFeatures()
    elif matrix_origin_type == 'random':
        compute_features = ComputeFeaturesRandom()

    feature_df = compute_features.get_feature_df(label_df)
    Utils.generate_relative_features(feature_df)
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
