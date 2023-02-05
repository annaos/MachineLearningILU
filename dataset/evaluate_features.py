import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from dataset.utils import Utils


def histogr(df_o, df_s):
    columns = Utils.get_features_list(collection)
    fig, ax = plt.subplots(3, 6, sharey='row', figsize=(25, 13))
    i = 0
    for feature_name in columns:
        df = pd.DataFrame().assign(original=df_o[feature_name], imitation=df_s[feature_name])
        ax_t = ax[i // 6, i % 6]

        df.astype(float).plot.hist(ax=ax_t, alpha=0.5, color=['red', 'blue'])
        ax_t.set_title(feature_name)
        i += 1
    plt.show()


def histogr1(df_o, df_s):
    feature_name = "density"
    fig, ax = plt.subplots(1, 1, figsize=(15, 20))

    n_bins=9
    df = pd.DataFrame().assign(original=df_o[feature_name], imitation=df_s[feature_name])
    ax_t = ax

    min_value = df.min().min()  # Get minimum value of column pairs, e.g. column 0 (a_wood) and column 3 (b_wood)
    max_value = df.max().max()  # Get maximum value of column pairs
    bins = np.linspace(min_value, max_value, n_bins)  # Create bins of equal size between min_value and max_value

    df.plot.hist(ax=ax_t, alpha=0.5, bins=bins, color=['red', 'blue'])
    ax_t.set_title(feature_name)

    plt.show()


def normalize(df):
    df = df.replace([True], 1).replace([False], 0)
    columns = Utils.get_features_list(collection)
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df

collection = 'pure'
#collection = 'relative'
df_o = pd.read_csv('../data/oct_dataset_original.csv')
df_o = df_o[df_o.rows < 10000]

df_s = pd.read_csv('../data/train_set_2_for_eval.csv')
#df_s = df_s.sample(n=len(df_o), ignore_index=True)
#histogr(normalize(df_o), normalize(df_s))
histogr(df_o, df_s)