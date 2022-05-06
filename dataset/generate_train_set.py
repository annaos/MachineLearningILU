import pandas as pd
import data_files

df = pd.read_csv(data_files.DATASET_PATH)

train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)

train_df.to_csv(data_files.TRAINSET_PATH, index=False)
test_df.to_csv(data_files.TESTSET_PATH, index=False)

print('Done')