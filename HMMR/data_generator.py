import numpy as np
import pandas as pd
from hmmr import *
from scipy.special import binom


def sample_x(j, n):
    if j == 1:
        return np.linspace(0.1, 10, num=n)
    elif j == 2:
        return np.log(np.linspace(0.1, 10, num=n))
    elif j == 3:
        return np.linspace(0.1, 10, num=n)**2
    elif j == 4:
        return np.random.normal(loc=1, scale=1.5, size=n)
    elif j == 5:
        return np.random.normal(loc=1, scale=1.5, size=n) ** 2
    else:
        raise NotImplementedError


def sample_input_file(k, n, m):
    cols = ['ID', 'Time'] + ['X{}'.format(i+1) for i in range(m)]
    all_seq_input = pd.DataFrame(columns=cols)

    for i in range(k):
        single_seq = pd.DataFrame(columns=cols)
        single_seq['Time'] = np.arange(n)
        single_seq['ID'] = i
        for j in range(m):
            single_seq['X{}'.format(j+1)] = sample_x(j+1, n)

        all_seq_input = all_seq_input.append(single_seq)

    all_seq_input = all_seq_input.round(decimals=2)
    return all_seq_input


def create_random_start_params(m, max_p, r):

    states = ['S{}'.format(i) for i in range(1, r+1)]
    inner_N = 20
    alpha = 0.1

    initial_probs = np.random.multinomial(inner_N, np.array([1]*r)/r)/inner_N
    initial_probs = pd.DataFrame([initial_probs], columns=states, index=['s'])

    transitions = np.zeros((r,r))

    for i in range(transitions.shape[0]):
        if i != 0:
            transitions[i, i - 1] = alpha
        if i != transitions.shape[0] - 1:
            transitions[i, i + 1] = alpha

        transitions[i, i] = 1 - 2*alpha

    transitions = pd.DataFrame(transitions, columns=states, index=states)
    regressions_dict = {}
    for s in states:
        sigma = 1
        p = max_p
        # p = np.random.random_integers(1, max_p+1)
        bin = int(binom(m+p, p))

        coef = list(np.random.uniform(-3, 4, size=bin))

        columns = ['sigma', 'p'] + ['b_{}'.format(i) for i in range(bin)]
        regression = pd.DataFrame([[sigma, p] + coef], columns=columns, index=[s])

        regression = regression.round(decimals=2)
        regressions_dict[s] = regression

    return initial_probs, transitions, regressions_dict


def sample_model_and_data(m, max_p, r, n, k):

    just_features = sample_input_file(k, n, m)
    initial_probs, transitions, emissions = create_random_start_params(m, max_p, r)

    sampler = HMMR(initial_probs, transitions, emissions)
    data = sampler.samples(just_features).round(decimals=2)

    return initial_probs, transitions, emissions, data