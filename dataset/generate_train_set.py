import pandas as pd
import data_files
from utils import Utils


df = pd.read_csv('/home/anna/Dokumente/KIT/Thesis/MachineLearningILU/data/dataset_split.csv')
df = Utils.update_is_effective(df, factor=1.5)
df = Utils.cut_df_by_is_effective(df, n=2000 * 2)

normalized_df = Utils.normalize(df)
train_df = normalized_df.sample(frac=0.8)
test_df = normalized_df.drop(train_df.index)

print("train len: ", len(train_df))
print("test len: ", len(test_df))

train_df.to_csv(data_files.TRAINSET_PATH, index=False)
test_df.to_csv(data_files.TESTSET_PATH, index=False)

print('Done')

