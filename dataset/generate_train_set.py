import pandas as pd
import data_files

def normalize(df):
    df = df.dropna().replace([True], 1).replace([False], 0)
    columns = ["rows", "nonzeros", "avg_nnz", "max_nnz", "std_nnz",
               "avg_row_block_count", "std_row_block_count", "min_row_block_count", "max_row_block_count",
               "avg_row_block_size", "std_row_block_size", "min_row_block_size", "max_row_block_size", "block_count"]
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df

df = pd.read_csv(data_files.DATASET_PATH)
df = df.sample(frac=0.1)

normalized_df = normalize(df)

train_df = normalized_df.sample(frac=0.8)
test_df = normalized_df.drop(train_df.index)

print("train len: ", len(train_df))
print("test len: ", len(test_df))

train_df.to_csv(data_files.TRAINSET_PATH, index=False)
test_df.to_csv(data_files.TESTSET_PATH, index=False)

print('Done')