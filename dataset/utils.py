import numpy as np
import pandas as pd
from scipy.stats import zscore


class Utils:
    @staticmethod
    def cut_df_by_is_effective(df, n=None):
        df1 = df[df.is_effective == 1]
        df0 = df[df.is_effective == 0]
        if n == None: halb = min(len(df1), len(df0))
        else: halb = min(round(n * 0.5), len(df1), len(df0))
        print("the half of set is ", halb)
        df = df1.sample(n=halb).append(df0.sample(n=halb))
        df = df.sample(frac=1, ignore_index=True)
        return df

    @staticmethod
    def cut_df_by_conv1(df, n=None):
        df1 = df[df.is_effective == 1]
        df0 = df[df.conv1 == 0]
        if n == None: halb = min(len(df1), len(df0))
        else: halb = min(round(n * 0.5), len(df1), len(df0))
        print("the half of set is ", halb)
        df = df1.sample(n=halb).append(df0.sample(n=halb))
        df = df.sample(frac=1, ignore_index=True)
        return df

    @staticmethod
    def update_is_effective(df, factor=1.5):
        df['is_effective'] = np.where((df['conv1'] == 1) & ((df['conv0'] == 0) | (df['relation'] > factor)), 1, 0)
        return df

    @staticmethod
    def generate_relative_features(df):
        columns = Utils.get_features('pure')
        feature_to_divide = Utils.get_features('rows').pop()
        for feature in columns:
            a = df[feature] / df[feature_to_divide]
            df[feature + '_relative'] = a
        return df


    @staticmethod
    def generate_zscore_features(df):
        columns = Utils.get_features('rows') + Utils.get_features('pure') + Utils.get_features('relative')
        for feature in columns:
            df[feature + '_zscore'] = (df[feature] - df[feature].mean()) / df[feature].std()
        return df

    @staticmethod
    def generate_normalize_features(df):
        df = df.replace([True], 1).replace([False], 0)
        columns = Utils.get_features('rows') + Utils.get_features('pure') + Utils.get_features('relative')
        for feature in columns:
            max_value = df[feature].max()
            min_value = df[feature].min()
            std = df[feature].std()
            if max_value != min_value:
                df[feature + '_normalized'] = (df[feature] - min_value) / (max_value - min_value)
            else:
                df[feature + '_normalized'] = df[feature]

            if std != 0:
                df[feature + '_zscore'] = (df[feature] - df[feature].mean()) / std
            else:
                df[feature + '_zscore'] = df[feature]
        return df

    @staticmethod
    def get_features(type = 'pure'):
        f = ['nonzeros', 'avg_nnz', 'max_nnz', 'std_nnz',
            'avg_row_block_count', 'std_row_block_count', 'min_row_block_count', 'max_row_block_count',
            'avg_row_block_size', 'std_row_block_size', 'min_row_block_size', 'max_row_block_size',
            'block_count']
        f_rows = ['rows']
        f_percent = ['posdef', 'psym', 'nsym', 'density']
        f_r = list(map(lambda x: x + '_relative', f))
        f_n = list(map(lambda x: x + '_normalized', f))
        f_r_n = list(map(lambda x: x + '_relative_normalized', f))
        f_z = list(map(lambda x: x + '_zscore', f))
        f_r_z = list(map(lambda x: x + '_relative_zscore', f))

        if type == 'pure':
            return f
        if type == 'percent':
            return f_percent
        if type == 'rows':
            return f_rows
        if type == 'relative':
            return f_r
        if type == 'normalized':
            return f_n
        if type == 'relative_normalized':
            return f_r_n
        if type == 'zscore':
            return f_z
        if type == 'relative_zscore':
            return f_r_z

    @staticmethod
    def get_features_list(type='all'):
        f = Utils.get_features('rows') + Utils.get_features('percent')
        if type == 'pure':
            return f + Utils.get_features('pure')
        if type == 'relative':
            return f + Utils.get_features('relative')
        if type == 'normalized':
            return f + Utils.get_features('normalized')
        if type == 'relative_normalized':
            return f + Utils.get_features('relative_normalized')
        if type == 'zscore':
            return f + Utils.get_features('zscore')
        if type == 'relative_zscore':
            return f + Utils.get_features('relative_zscore')
        if type == 'all':
            return f + Utils.get_features('relative') + Utils.get_features('normalized') + Utils.get_features('relative_normalized')

def filter_original_set(df):
    original_df =  pd.read_csv('/home/anna/Dokumente/KIT/Thesis/MachineLearningILU/data/dataset_original.csv')
    original_problems = original_df.problem_id
    a = df['id'].isin(original_problems)
    filtered_df = df[a]
    right_df = df.drop(filtered_df.index)
    return right_df
