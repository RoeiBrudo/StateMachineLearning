import pandas as pd
import os
from hmmr import *
from baum_welsh import *
from data_generator import create_random_start_params
from scipy.special import binom


def learn_hmm_based_on_estimators(initial_probs, transitions, regressions, data, n_start, coef_noise):

    if n_start == 0:
        start_hmm = HMMR(initial_probs, transitions, regressions)
        learned_hmm, ll = baum_welsh(start_hmm, data)
        return learned_hmm

    else:
        hmms = []
        ll_of_all_data = []
        for s in range(n_start):

            # New regressions start point
            new_regressions = copy.deepcopy(regressions)
            for regression_key in new_regressions.keys():
                regression = regressions[regression_key]
                p = regression['p']
                m = len(get_columns_of_data(data))
                bin = int(binom(m+p, p))

                coef_noise = (np.random.uniform(-coef_noise, coef_noise, size=bin))
                regression.iloc[:, 2:] = regression.iloc[:, 2:] + coef_noise

            new_transitions = copy.deepcopy(transitions)
            for s in range(new_transitions.shape[0]):
                max_noise = np.min(new_transitions.iloc[s].values[new_transitions.iloc[s].values > 0])
                noise = np.random.uniform(0, max_noise)
                if np.random.random() < 0.5:
                    noise = -noise
                if s == 0:
                    new_transitions.iloc[s, s + 1] = new_transitions.iloc[s, s + 1] + noise
                    new_transitions.iloc[s, s] = new_transitions.iloc[s, s] - noise
                elif s == new_transitions.shape[0] - 1:
                    new_transitions.iloc[s, s - 1] = new_transitions.iloc[s, s - 1] + noise
                    new_transitions.iloc[s, s] = new_transitions.iloc[s, s] - noise
                else:
                    new_transitions.iloc[s, s + 1] = new_transitions.iloc[s, s + 1] + noise
                    new_transitions.iloc[s, s - 1] = new_transitions.iloc[s, s - 1] + noise
                    new_transitions.iloc[s, s] = new_transitions.iloc[s, s] - 2 * noise

            # learn with noisy estimators
            start_hmm = HMMR(initial_probs, new_transitions, new_regressions)
            learned_hmm, ll = baum_welsh(start_hmm, data)
            ll_of_all_data.append(ll)
            hmms.append(learned_hmm)

        argmax_ll = int(np.argmax(ll_of_all_data))
        print(ll_of_all_data)
        print(argmax_ll)
        return hmms[argmax_ll]


def learn_hmm_random_starts(n_states_range, p_range, coef_range, transitions_alpha, n_starts, data):
    m = len(get_columns_of_data(data))
    hmms = []
    ll_of_all_data = []
    for start in range(n_starts):

        if n_states_range[0] == n_states_range[1]:
            n_states = n_states_range[0]
        else:
            n_states = np.random.random_integers(n_states_range[0], n_states_range[1])

        if p_range[0] == p_range[1]:
            p = p_range[0]
        else:
            p = np.random.random_integers(p_range[0], p_range[1])

        initial_probs, transitions, regressions = get_single_start_params(n_states, m, p, coef_range, transitions_alpha)
        start_hmm = HMMR(initial_probs, transitions, regressions)
        learned_hmm, ll = baum_welsh(start_hmm, data)
        ll_of_all_data.append(ll)
        hmms.append(learned_hmm)

    argmax_ll = int(np.argmax(ll_of_all_data))

    return hmms[argmax_ll]


def get_single_start_params(r, m, p, coef_range, transitions_alpha):
    states = ['S{}'.format(i) for i in range(1, r+1)]
    inner_N = 20
    alpha = np.random.uniform(0.5 * transitions_alpha, 1.5 * transitions_alpha)

    initial_probs = np.random.multinomial(inner_N, np.array([1]*r)/r)/inner_N
    initial_probs = pd.DataFrame([initial_probs], columns=states, index=['s'])

    transitions = np.zeros((r, r))

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
        bin = int(binom(m+p, p))

        coef = list(np.random.uniform(coef_range[0], coef_range[1], size=bin))

        columns = ['sigma', 'p'] + ['b_{}'.format(i) for i in range(bin)]
        regression = pd.DataFrame([[sigma, p] + coef], columns=columns, index=[s])

        regression = regression.round(decimals=2)
        regressions_dict[s] = regression

    return initial_probs, transitions, regressions_dict


def get_params_from_dir(initial_params_dir):
    initial_probs = pd.read_csv(initial_params_dir + '/initial_prob.csv', index_col='s')
    transitions = pd.read_csv(initial_params_dir + '/transitions.csv', index_col='s')
    regression_files = [f for f in os.listdir(initial_params_dir) if f.startswith('regression')]
    regressions = {}
    for file in regression_files:
        emissions = pd.read_csv(initial_params_dir + '/' + file, index_col='s')
        state = emissions.index.unique()[0]
        regressions[state] = emissions

    return initial_probs, transitions, regressions
