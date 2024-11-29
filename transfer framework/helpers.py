import numpy as np
import pandas as pd


def split_to_batches_full(data_df, batch_size=32):
    n_batches = data_df.shape[0] // 32

    ind = np.arange(data_df.shape[0])
    np.random.shuffle(ind)
    data_df_copy = data_df.copy()
    data_df_copy = data_df_copy.iloc[ind]

    Xs, Ys, Ks = [], [], []

    for i in range(n_batches + 1):
        if i * batch_size < data_df.shape[0]:
            batched_data = data_df_copy.iloc[i * batch_size:i * batch_size + batch_size]
            Xs.append(batched_data.iloc[:, :-2].values)
            Ys.append(batched_data.iloc[:, -2].values)
            Ks.append(batched_data.iloc[:, -1].values)

    return Xs, Ys, Ks


def reduce_data(df, k, n_single):
    list_of_dfs = [df[df.k == i].iloc[:n_single, :] for i in range(1, k+1)]
    df = pd.concat(list_of_dfs)
    df.sample(frac=1)
    return df
