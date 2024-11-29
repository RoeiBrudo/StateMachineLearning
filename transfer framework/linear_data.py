import numpy as np
import pandas as pd


def get_classifier(W, theta):
    def classify(x):
        return np.sign(x @ W + theta)

    return classify


def generate_true_random_classifiers(d, k):
    """
    d - input dimension
    k - number of classifiers
    """

    W = np.random.uniform(low=-1, high=1, size=d)
    thetas = np.random.uniform(low=-3, high=3, size=k)

    classifiers = []
    for k_i in range(thetas.shape[0]):
        classifier = get_classifier(W, thetas[k_i])
        classifiers.append(classifier)

    return classifiers


def flip_array_classification(arr, noise_param):
    ind = np.random.choice(np.arange(arr.shape[0]), p=[noise_param]*arr.shape[0])
    arr[ind] *= -1
    return arr


def create_data(d, k, classifiers, n, noise=None):

    data_df = pd.DataFrame(np.zeros((n, d + 2)), columns=[f'x_{i}' for i in range(1, d + 1)] + ['y', 'k'])
    data_df.iloc[:, :d] = np.random.uniform(low=-4, high=4, size=(n, d))
    ind = np.arange(n)
    n_single = n // k
    ind_list = [ind[n_single * i:n_single * i + n_single] for i in range(k)]
    data_df = data_df.iloc[:n_single*k]

    for k_i in range(k):
        classifier = classifiers[k_i]
        data_df.iloc[ind_list[k_i], d + 1] = k_i
        data_df.iloc[ind_list[k_i], d] = classifier(data_df.iloc[ind_list[k_i], :d].copy())

    data_df.iloc[:, d + 1] = data_df.iloc[:, d + 1].astype(np.int)
    data_df = data_df.sample(frac=1)

    if noise is not None:
        data_df.iloc[:int(noise*data_df.shape[0]), -2] *= -1
        data_df = data_df.sample(frac=1)

    return data_df


