try:
    import os
    from compute_features import get_feature_df
    # from benchmark import Benchmark
    import pandas as pd

    DATA_PATH = './data/'
    MATRICES_PATH = DATA_PATH + 'matrices.csv'
    DATASET_PATH = DATA_PATH + 'dataset.csv'
    META_PATH = DATA_PATH + 'matrices_meta.json'

    label_df = pd.read_csv(MATRICES_PATH)

    print("got label df")
    feature_df = get_feature_df()
    print("got feature df")

    dataset = label_df.merge(feature_df, left_on="ProblemId", right_on="id")
    print("merged dfs")

    dataset.to_csv(DATA_PATH + "dataset.csv", index=False)
    print("saved dataset")
except Exception as e:
    print(e)
