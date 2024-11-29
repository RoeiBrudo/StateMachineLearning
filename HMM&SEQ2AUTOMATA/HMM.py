from hmmlearn.hmm import GaussianHMM
import itertools
from discretizer import *


class HmmLerner:
    def __init__(self, scaled_data_set,  predict_column,
                 N_test, n_hidden_states):

        self.N_test = N_test
        self.n_hidden_states = n_hidden_states

        self.pre_column = predict_column
        self.scaled_data = scaled_data_set.copy()
        self.pre_column_index = list(self.scaled_data.columns).index(self.pre_column)
        self.hmm = GaussianHMM(n_components=self.n_hidden_states)
        self.hmm.fit(self.scaled_data.iloc[0:self.scaled_data.shape[0]-self.N_test])
        # self.empty_prediction = self.get_empty_prediction()

        self.prediction_states = self.hmm.means_[:,self.pre_column_index]

    def test(self, input_data=None):
        if input_data is None:
            data = self.scaled_data
        else:
            data = input_data

        predicted_scaled = []
        d = 0
        for day in (range(data.shape[0] - self.N_test, data.shape[0])):
            print('calculate day ', d)
            d += 1
            cur_data = data[0:day].values
            predicted_scaled.append(self.predict(cur_data))

        predicted_scaled = np.array(predicted_scaled)
        predicted_scaled = predicted_scaled.reshape(predicted_scaled.shape[0], 1)
        self.predict_by_model_scaled = predicted_scaled
        return predicted_scaled

    def predict(self, X):
        if X == []:
            states_prob = self.hmm.startprob_
        else:
            states_prob = self.hmm.predict_proba(X)[-1]

        prediction = np.dot(states_prob, self.prediction_states)

        # most_probable_state = np.argmax(np.array(states_prob))
        # out = self.hmm.means_[most_probable_state]

        # prediction = out[self.pre_column_index]
        return prediction










    # self.steps_to_check_pre = steps_to_check_pre
    # self.range_to_check_pre = range_to_check_pre
    # self.steps_to_check_feat = steps_to_check_feat
    # self.range_to_check_feat = range_to_check_feat

    # def get_empty_prediction(self):
    #     possible_outcomes = self.get_all_possible_outcomes(self.scaled_data.columns)
    #     data = pd.DataFrame(data=[], columns=self.scaled_data.columns)
    #     outcome_scores = []
    #     for possible_outcome in possible_outcomes:
    #         next_pre = pd.DataFrame(data=[possible_outcome], columns=self.scaled_data.columns)
    #         total_data = pd.concat([data, next_pre], sort=True)
    #         s = self.hmm.score(total_data)
    #         outcome_scores.append(s)
    #
    #     most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)][
    #                                                         list(self.scaled_data.columns).index(self.pre_column)]
    #     return most_probable_outcome

    # def get_all_possible_outcomes(self, columns, previous_day=None, cur_day=None):
    #     if previous_day is None:
    #         ranges = [np.linspace(0, 1, 10)] * len(columns)
    #         return np.array(list(itertools.product(*ranges)))
    #
    #     ranges = []
    #     if cur_day is None:
    #         reference = previous_day
    #     else:
    #         # reference = previous_day
    #         reference = cur_day
    #         # reference[list(columns).index(self.pre_column)] = previous_day[list(columns).index(self.pre_column)]
    #
    #     for i, column in enumerate(columns):
    #         if column == self.pre_column:
    #             dist = self.range_to_check_pre
    #             steps_to_check = self.steps_to_check_pre
    #         else:
    #             dist = self.range_to_check_feat
    #             steps_to_check = self.steps_to_check_feat
    #
    #         minimum = max(reference[i] - dist, 0)
    #         maximum = min(reference[i] + dist, 1)
    #         rang = np.linspace(minimum, maximum, steps_to_check)
    #         ranges.append(rang)
    #
    #     possible_outcomes = np.array(list(itertools.product(*ranges)))
    #     return possible_outcomes

    # def get_most_probable_outcome(self, input_data, cur_day_given=True):
    #     l = len(input_data)
    #     if l == 0:
    #         return self.empty_prediction
    #     elif cur_day_given:
    #         last_day = l - 1
    #         possible_outcomes = self.get_all_possible_outcomes(self.scaled_data.columns,
    #                                                            input_data.loc[l-2], input_data.loc[l-1])
    #         data = self.scaled_data[0:last_day]
    #     else:
    #         last_day = l
    #         possible_outcomes = self.get_all_possible_outcomes(self.scaled_data.columns, input_data[l-1])
    #         data = self.scaled_data[0:last_day]
    #
    #     outcome_scores = []
    #     for possible_outcome in possible_outcomes:
    #         next_pre = pd.DataFrame(data=[possible_outcome], columns=self.scaled_data.columns)
    #         total_data = pd.concat([data, next_pre], sort=True)
    #         s = self.hmm.score(total_data)
    #         outcome_scores.append(s)
    #
    #     most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)][
    #                                                         list(self.scaled_data.columns).index(self.pre_column)]
    #     return most_probable_outcome
