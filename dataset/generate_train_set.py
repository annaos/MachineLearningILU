import pandas as pd
from dataset.utils import Utils

DATA_PATH = '../data/'
DATASET_PATH = DATA_PATH + 'dataset_original_up_to_100Trows.csv'
TRAINSET_PATH = DATA_PATH + 'train_set_original_part.csv'
TESTSET_PATH = DATA_PATH + 'test_set_original_part.csv'

def generate_train():
    df = pd.read_csv(DATASET_PATH)
    # df = Utils.generate_relative_features(df)
    # df = Utils.cut_df_by_is_effective(df, n=20000 * 2)

    normalized_df = Utils.generate_normalize_features(df).generate_zscore_features(df)
    train_df = normalized_df.sample(frac=0.8)
    test_df = normalized_df.drop(train_df.index)

    print("train len: ", len(train_df))
    print("test len: ", len(test_df))

    train_df.to_csv(TRAINSET_PATH, index=False)
    test_df.to_csv(TESTSET_PATH, index=False)

    print('Done')


def generate_test_mini():
    df = pd.read_csv(DATA_PATH + 'dataset_original_up_to_100Trows.csv')
    df = Utils.cut_df_by_conv1(df, n=5 * 2)
    normalized_df = Utils.generate_normalize_features(df)
    train_df = normalized_df.sample(frac=1)
    print("train len: ", len(train_df))

    train_df = train_df.sort_values(by=['problem_id'], ignore_index=True)
    train_df.to_csv(DATA_PATH + 'test_set_original_mini.csv', index=False)
    print('Done')


if __name__ == "__main__":
    generate_train()
    # generate_test_mini()
