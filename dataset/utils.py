import numpy as np


class Utils:
    @staticmethod
    def cut_df_by_is_effective(df, n=100):
        halb = round(n * 0.5)
        df1 = df[df.isEffective == 1]
        df0 = df[df.isEffective == 0]
        halb = min(halb, len(df1), len(df0))
        df = df1.sample(n=halb).append(df0.sample(n=halb))
        df = df.sample(frac=1, ignore_index=True)
        return df

    @staticmethod
    def update_is_effective(df, factor=1.5):
        df['isEffective'] = np.where((df['conv1'] == 1) & ((df['conv0'] == 0) | (df['relation'] > factor)), 1, 0)
        return df

    @staticmethod
    def generate_relative_features(df):
        feature_list_to_relative = ['nonzeros',
            'avg_nnz', 'max_nnz', 'std_nnz',
            'avg_row_block_count', 'std_row_block_count', 'min_row_block_count',
            'max_row_block_count',
            'avg_row_block_size', 'std_row_block_size', 'min_row_block_size',
            'max_row_block_size',
            'block_count']
        feature_to_divide = 'rows'
        for feature in feature_list_to_relative:
            df[feature + '_relative'] = df[feature] / df[feature_to_divide]
        return df
