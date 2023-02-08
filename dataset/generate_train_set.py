import pandas as pd
from dataset.utils import Utils

DATA_PATH = '../data/'
DATASET_PATH = DATA_PATH + 'dataset_original_up_to_100Trows.csv'
TRAINSET_PATH = DATA_PATH + 'train_set_original_balance_part_2.csv'
TESTSET_PATH = DATA_PATH + 'test_set_original_balance_part_2.csv'

def generate_train():
    df = pd.read_csv(DATASET_PATH)
    # df = Utils.generate_relative_features(df)
    df = Utils.cut_df_by_is_effective(df, n=20000 * 2)

    df = Utils.generate_normalize_features(df)
    # train_df = df.sample(frac=0.8)
    # test_df = df.drop(train_df.index)

    df1 = df[df.is_effective == 1]
    df0 = df[df.is_effective == 0]

    train_df1 = df1.sample(frac=0.8)
    test_df1 = df1.drop(train_df1.index)

    train_df0 = df0.sample(frac=0.8)
    test_df0 = df0.drop(train_df0.index)

    train_df = train_df1.append(train_df0).sample(frac=1, ignore_index=True)
    test_df = test_df1.append(test_df0).sample(frac=1, ignore_index=True)

    print("train len: ", len(train_df))
    print("train len eff1: ", len(train_df[train_df.is_effective == 1]))
    print("train len eff0: ", len(train_df[train_df.is_effective == 0]))
    print("test len: ", len(test_df))
    print("test len eff1: ", len(test_df[test_df.is_effective == 1]))
    print("test len eff0: ", len(test_df[test_df.is_effective == 0]))

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
