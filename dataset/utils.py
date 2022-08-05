import numpy as np


class Utils:
    @staticmethod
    def cut_df_by_is_effective(df, n=100):
        halb = round(n * 0.5)
        df1 = df[df.isEffective == 1]
        df0 = df[df.isEffective == 0]
        halb = min(halb, len(df1), len(df0))
        print(halb)
        print("the half of set is ", halb)
        df = df1.sample(n=halb).append(df0.sample(n=halb))
        df = df.sample(frac=1, ignore_index=True)
        return df

    @staticmethod
    def update_is_effective(df, factor=1.5):
        df['isEffective'] = np.where((df['conv1'] == 1) & ((df['conv0'] == 0) | (df['relation'] > factor)), 1, 0)
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
    def normalize(df):
        df = df.dropna().replace([True], 1).replace([False], 0)
        columns = Utils.get_features('pure') + Utils.get_features('relative')
        for feature in columns:
            max_value = df[feature].max()
            min_value = df[feature].min()
            if (max_value != min_value):
                df[feature + '_normalized'] = (df[feature] - min_value) / (max_value - min_value)
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

    @staticmethod
    def get_features_list(type='all'):
        f = Utils.get_features('rows') + Utils.get_features('percent')
        if type == 'original':
            return f + Utils.get_features('pure')
        if type == 'relative':
            return f + Utils.get_features('relative')
        if type == 'normalized':
            return f + Utils.get_features('normalized')
        if type == 'relative_normalized':
            return f + Utils.get_features('relative_normalized')
        if type == 'all':
            return f + Utils.get_features('relative') + Utils.get_features('normalized') + Utils.get_features('relative_normalized')

