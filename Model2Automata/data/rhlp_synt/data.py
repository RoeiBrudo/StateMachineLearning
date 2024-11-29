import numpy as np
import itertools
from scipy.special import softmax
import pandas as pd


def get_train_test_data():
    file = 'data/rhlp_synt/rhlp_data.csv'
    df = pd.read_csv(file)
    # df_2d = df.loc[[col for col in df.columns if col != ['ID', 'Time']]]
    train_ids = df.ID.unique()[:np.int(0.7*df.ID.unique().size)]
    test_ids = df.ID.unique()[np.int(0.7*df.ID.unique().size):]
    train_X = []
    test_X = []
    train_y = []
    test_y = []

    for id in train_ids:
        train_X.append(df[df.ID == id].drop(columns=['ID', 'Time', 'y']).values)
        train_y.append(df[df.ID == id]['y'].values)

    for id in test_ids:
        test_X.append(df[df.ID == id].drop(columns=['ID', 'Time', 'y']).values)
        test_y.append(df[df.ID == id]['y'].values)

    return train_X, train_y, test_X, test_y
