import numpy as np
import itertools
from scipy.special import softmax
import pandas as pd


def generate_data_df(m, h, n):
    reg_coef = np.random.uniform(-1, 1, (h, m + 1))
    logistic_coef = np.random.uniform(-1, 1, (h, m + 1))

    columns = ['ID', 'Time'] + ['X{}'.format(i) for i in range(m)] + ['y']
    final_df = pd.DataFrame([], columns=columns)
    IDs = 100
    for i in range(IDs):
        cur_n = np.int(n / IDs)

        if i == IDs - 1:
            cur_n = n - (IDs - 1) * np.int(n / IDs)

        cur_x = np.array([np.random.uniform(-50, 50, cur_n) for _ in range(m)] + [[1]*cur_n]).T
        cur_y = []

        for j in range(cur_x.shape[0]):
            h_logistic = np.dot(logistic_coef, cur_x[j])
            hidden = softmax(h_logistic)

            all_regressions = np.dot(reg_coef, cur_x[j])
            y = np.dot(all_regressions, hidden)
            cur_y.append(y)

        cur_y = np.array(cur_y)
        cur_y += np.random.normal(0, 1, cur_y.shape)

        cur_df = pd.DataFrame(np.zeros((cur_n, m+3)), columns=columns)
        cur_df['ID'] = i
        cur_df['Time'] = np.arange(cur_n)
        cur_df[['X{}'.format(i) for i in range(m)]] = cur_x[:, :-1]
        cur_df['y'] = cur_y

        final_df = final_df.append(cur_df)

    final_df.astype(np.float)
    final_df.to_csv('rhlp_data.csv', index=False)


def get_train_test_data():
    file = 'train_data/rhlp_data.csv'
    df = pd.read_csv(file)
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


if __name__ == '__main__':
    generate_data_df(6, 5, 10000)