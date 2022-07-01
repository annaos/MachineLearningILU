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
        if (max_value != min_value):
            df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df

def ret_df_isEffective(table, part = None, n = 100):
    qwt_out = n if part == None else int(round(len(table) * part))
    qwt1 = qwt0 = int(round(qwt_out * 0.5))
    table1 = table[table.isEffective == 1]
    if len(table1) < qwt1:
        print('Not enough 1')
        qwt1 = len(table1)
    table = table[table.isEffective == 0]
    if len(table) < qwt0:
        print('Not enough 0')
        qwt0 = len(table)
    print(f'Amount effective records: {qwt1}, not effective records: {qwt0}')
    table1 = table1.sample(n=qwt1)
    table = table.sample(n=qwt0)
    table = table.append(table1)
    table = table.sample(frac=1)
    return table

df = pd.read_csv(data_files.DATASET_PATH)
df = ret_df_isEffective(df, n=500*2)

normalized_df = normalize(df)
train_df = normalized_df.sample(frac=0.8)
test_df = normalized_df.drop(train_df.index)

print("train len: ", len(train_df))
print("test len: ", len(test_df))

train_df.to_csv(data_files.TRAINSET_PATH, index=False)
test_df.to_csv(data_files.TESTSET_PATH, index=False)

print('Done')