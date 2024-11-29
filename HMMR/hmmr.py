from inner_models.regression import *
from scipy.special import logsumexp


class HMMR:
    def __init__(self, initial_probs, init_transitions, init_emissions, from_regressions=False):
        self.states = initial_probs.columns
        self.nstates = len(self.states)
        self.initial_probs = np.array(initial_probs.iloc[0])
        self.transitions = init_transitions
        if from_regressions:
            self.set_regressions(init_emissions)
        else:
            self.init_emissions_regression(init_emissions)

        with np.errstate(divide='ignore'):
            self.log_transitions = np.log(self.transitions)
            self.log_initial_probs = np.log(self.initial_probs)

    def init_emissions_regression(self, init_regressions):

        self.regressions_models = {}
        for state in init_regressions.keys():
            regression_model = Regression(init_regressions[state])
            self.regressions_models[state] = regression_model

    def set_regressions(self, regressions):
        self.regressions_models = regressions

    def sample(self, input_data):

        state_index = np.argmax(np.random.multinomial(1, self.initial_probs))
        cur_state = self.states[state_index]
        states = []
        outputs = np.array([0.0]*input_data.shape[0])
        for _ in range(input_data.shape[0]):
            states.append(cur_state)
            state_index = np.argmax(np.random.multinomial(1, self.transitions.loc[cur_state]))
            cur_state = self.states[state_index]

        states = np.array(states)
        data = input_data[[col for col in input_data.columns if col not in ['ID', 'Time']]]
        for state in self.regressions_models.keys():
            state_df = data[states == state]
            output = self.regressions_models[state].predict_noisy(state_df)
            outputs[states == state] = output.flatten()

        return states, outputs

    def samples(self, input_data):
        final_data = pd.DataFrame(input_data, copy=True)
        final_data.insert(loc=len(final_data.columns), column='Y', value=[0]*final_data.shape[0])
        final_data.insert(loc=len(final_data.columns), column='state', value=[0]*final_data.shape[0])
        for id in input_data.ID.unique():
            states, outputs = self.sample(input_data[input_data.ID == id])
            final_data.loc[final_data.ID == id, 'state'] = states
            final_data.loc[final_data.ID == id, 'Y'] = outputs

        return final_data

    def viterbi(self, in_seq, y_seq):
        N_samples = len(in_seq)

        log_alpha = np.zeros((len(self.states), N_samples), dtype=np.float)
        M_table = np.zeros((len(self.states), N_samples), dtype=np.int)

        only_emissions_prob = self.__only_log_probabilities_values__(in_seq, y_seq)

        start_column = self.log_initial_probs + only_emissions_prob[:, 0]
        log_alpha[:, 0] = start_column[0]

        for i in range(1, N_samples):
            for j, state in enumerate(self.states):
                p_to_state = log_alpha[:, i - 1] + self.log_transitions[state].values
                argmax_p_to_state = np.argmax(p_to_state)
                M_table[j, i] = argmax_p_to_state
                log_alpha[j, i] = p_to_state[argmax_p_to_state] + only_emissions_prob[j, i]


        # get MAP
        predicted_states = []
        cur_state_index = np.argmax(log_alpha[:, len(in_seq) - 1])
        predicted_states.append(self.states[cur_state_index])
        for pos_i in range(N_samples - 1, 0, -1):
            cur_state_index = M_table[cur_state_index, pos_i]
            predicted_states.append(self.states[cur_state_index])

        predicted_states.reverse()
        return predicted_states

    def __forward__(self, in_seq, y_seq):

        log_alpha = np.zeros((len(self.states), len(in_seq)), dtype=np.float)

        only_emissions_prob = self.__only_log_probabilities_values__(in_seq, y_seq)

        start_column = self.log_initial_probs + only_emissions_prob[:, 0]
        log_alpha[:, 0] = start_column[0]

        for i in range(1, len(in_seq)):
            for j, state in enumerate(self.states):
                p_to_state = log_alpha[:, i - 1] + self.log_transitions[state].values
                p_to_state = logsumexp(p_to_state)
                log_alpha[j, i] = p_to_state + only_emissions_prob[j, i]

        return log_alpha

    def __backward__(self, in_seq, y_seq):
        log_beta = np.zeros((len(self.states), len(in_seq)), dtype=np.float64)

        only_emissions_prob = self.__only_log_probabilities_values__(in_seq, y_seq)

        for t in range(len(in_seq) - 2, -1, -1):
            next_probabilities = log_beta[:, t + 1]

            for j, state in enumerate(self.states):
                transitions = self.log_transitions.loc[state].values

                multiplications = transitions + next_probabilities + only_emissions_prob[:, t+1]
                log_beta[j, t] = logsumexp(multiplications)

        return log_beta

    def __only_log_probabilities_values__(self, in_seq, y_seq):
        only_emissions_prob = []
        for j, state in enumerate(self.states):
            probabilities = self.regressions_models[state].get_log_prob_of_seq(in_seq, y_seq)
            only_emissions_prob.append(probabilities)
        only_emissions_prob = np.array(only_emissions_prob)
        return only_emissions_prob

    def get_ll_of_data(self, in_seq, y_seq):
        log_alpha = self.__forward__(in_seq, y_seq)
        seq_ll = logsumexp(log_alpha[:, log_alpha.shape[1]-1])
        return seq_ll

    def get_ll_of_all_data(self, in_sequences, y_sequences):
        seq_ll = []
        for in_seq, y_seq in zip(in_sequences, y_sequences):
            seq_ll.append(self.get_ll_of_data(in_seq, y_seq))

        return logsumexp(seq_ll)

    def write_files(self, results_folder):
        init_prob_df = pd.DataFrame(self.initial_probs.reshape((1, len(self.states))), columns=self.states, index=['s'])
        init_prob_df.to_csv(results_folder + '/initial_prob.csv', index_label='s')
        self.transitions.to_csv(results_folder + '/transitions.csv', index_label='s')

        for id in self.transitions.index:
            regression_data = self.regressions_models[id].get_regression_for_file()
            columns = ['sigma', 'p'] + ['b_{}'.format(i) for i in range(len(regression_data['coef']))]
            emissions = pd.DataFrame([[regression_data['sigma'], regression_data['p']] + regression_data['coef']],
                                     columns=columns, index=[id])

            emissions = emissions.round(decimals=2)
            emissions.to_csv(results_folder + '/regression_{}.csv'.format(id), index_label='s')
