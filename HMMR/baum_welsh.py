import numpy as np
import copy
from hmmr import *
from sklearn.preprocessing import normalize


def get_columns_of_data(data, time=False):
    if time:
        columns = [col for col in data.columns if col not in ['ID', 'state', 'Y']]
    else:
        columns = [col for col in data.columns if col not in ['ID', 'Time', 'state', 'Y']]

    return columns


def pre_process_to_hmmr(data, time=False):
    sequences = [data[data.index == i] for i in data.index.unique()]
    columns = get_columns_of_data(data, time=time)
    in_sequences = [seq[columns] for seq in sequences]
    y_sequences = [seq['Y'] for seq in sequences]
    return in_sequences, y_sequences


def EM_step(cur_HMM, in_sequences, y_sequences):
        N_transitions = np.zeros((len(cur_HMM.states), len(cur_HMM.states), len(in_sequences)))
        N_start = np.zeros((len(cur_HMM.states), len(in_sequences)))

        all_in_seq = pd.concat(in_sequences)
        all_y_seq = pd.concat(y_sequences)
        all_log_gammas = np.zeros((cur_HMM.nstates, all_in_seq.shape[0]), dtype=np.float)
        last_index = 0

        for i, (in_seq, y_seq) in enumerate(zip(in_sequences, y_sequences)):
            log_alpha, log_beta = cur_HMM.__forward__(in_seq, y_seq), cur_HMM.__backward__(in_seq, y_seq)
            seq_likelihood = logsumexp(log_alpha[:, log_alpha.shape[1] - 1])  # sequence log probability
            log_gamma = log_alpha + log_beta - seq_likelihood  # log posterior probabilities

            # N_hat for initial probabilities
            N_start[:, i] = log_gamma[:, 0]

            # N_hat for transitions
            N_transitions_seq = get_transitions_n_hat_for_single_seq(cur_HMM, log_alpha, log_beta, in_seq, y_seq)
            N_transitions[:, :, i] = N_transitions_seq

            new_last = last_index + log_gamma.shape[1]
            all_log_gammas[:, last_index:new_last] = log_gamma
            last_index = new_last

        # # Initial probabilities update
        N_all_start = logsumexp(N_start, axis=1).reshape((1, len(cur_HMM.states)))
        N_start = pd.DataFrame(N_all_start, columns=cur_HMM.states, index=['Start'])
        norm_start_vector = logsumexp(N_start)
        updated_log_initial_probs = N_start.subtract(norm_start_vector)
        new_initial_probs = pd.DataFrame(np.exp(updated_log_initial_probs))

        # transition probabilities update
        N_all_transitions = logsumexp(N_transitions, axis=2)
        N_transitions = pd.DataFrame(N_all_transitions, columns=cur_HMM.states, index=cur_HMM.states)
        norm_transitions_vector = logsumexp(N_transitions, axis=1)
        updated_log_transitions = N_transitions.subtract(norm_transitions_vector, axis=0)
        new_transitions = pd.DataFrame(np.exp(updated_log_transitions))


        new_regressions = copy.deepcopy(cur_HMM.regressions_models)

        for i, state in enumerate(cur_HMM.states):

            weights = all_log_gammas[i, :]

            new_regressions[state].update_regression(all_in_seq, all_y_seq, weights)

        return new_initial_probs, new_transitions, new_regressions


def get_transitions_n_hat_for_single_seq(cur_HMM, log_alpha, log_beta, in_seq, y_seq):
        """
        :param log_alpha: log forward table
        :param log_beta: log backward table
        :param log_emissions: log emissions table
        :param log_transitions: log transitions table
        :param seq: the seq relשed to log alpha and log beta
        :return: transitions empirical counts for the given sequence (weighted with their probabilities). Log-space
        """
        log_seq_likelihood = logsumexp(log_alpha[:, log_alpha.shape[1]-1])
        N_transitions = np.zeros((len(cur_HMM.states), len(cur_HMM.states), len(in_seq)-1)) - np.inf
        emissions = cur_HMM.__only_log_probabilities_values__(in_seq, y_seq)

        for t in range(1, len(in_seq)):
            alpha_col = log_alpha[:, t-1]
            beta_col = log_beta[:, t]

            emissions_col = emissions[:, t]

            emissions_mat = np.broadcast_to(emissions_col, (len(alpha_col), len(alpha_col)))
            alpha_mat = np.broadcast_to(alpha_col, (len(alpha_col), len(alpha_col))).T
            beta_mat = np.broadcast_to(beta_col, (len(beta_col), len(beta_col)))

            N_transitions[:, :, t-1] = (alpha_mat + beta_mat + emissions_mat + cur_HMM.log_transitions)\
                                       - log_seq_likelihood

        states_to_state = logsumexp(N_transitions, axis=2)

        return states_to_state


def baum_welsh(start_hmm, data):
    in_sequences, y_sequences = pre_process_to_hmmr(data)

    diff = np.inf
    cur_HMM = start_hmm
    ll = cur_HMM.get_ll_of_all_data(in_sequences, y_sequences)
    print('start ll = ', ll)
    iteration = 0

    while diff > 0.1:
        iteration += 1

        new_initial_probs, new_transitions, new_regressions = EM_step(cur_HMM, in_sequences, y_sequences)
        new_HMM = HMMR(new_initial_probs, new_transitions, new_regressions, from_regressions=True)

        new_ll = new_HMM.get_ll_of_all_data(in_sequences, y_sequences)
        diff = new_ll - ll
        print('diff', diff)
        if diff > 0:
            ll = new_ll
            cur_HMM = new_HMM

    print("init_prob: ", cur_HMM.initial_probs)
    print("transitions: ", cur_HMM.transitions)
    for state in cur_HMM.states:
        print(state, 'regression:')
        print('coef', cur_HMM.regressions_models[state].regression.coef_)
        print('sigma', cur_HMM.regressions_models[state].sigma)
        print('p', cur_HMM.regressions_models[state].p)
    return cur_HMM, ll
