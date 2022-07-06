import numpy as np

class Utils:
    def cut_df_by_is_effective(df,  n = 100):
        halb = round(n * 0.5)
        df1 = df[df.isEffective == 1]
        df0 = df[df.isEffective == 0]
        halb = min(halb, len(df1), len(df0))
        df = df1.sample(n=halb).append(df0.sample(n=halb))
        df = df.sample(frac=1, ignore_index=True)
        return df

    def update_is_effective(df, factor = 1.5):
        df['isEffective'] = np.where((df['conv1'] == 1) & ((df['conv0'] == 0) | (df['relation'] > factor)), 1, 0)
        return df
