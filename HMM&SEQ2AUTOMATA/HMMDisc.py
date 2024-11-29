from discretizer import *
from HMM import *
import csv


class HmmDisc:

    def __init__(self, scaled_data_set, predict_column, features_clusters=15, prediction_clusters=10, hmm_lerner=None):
        self.pre_column = predict_column
        self.scaled_data = scaled_data_set
        if hmm_lerner is None:
            self.hmm_learner = HmmLerner(scaled_data_set, self.pre_column,  4, 25, 0.15, 50)
        else:
            self.hmm_learner = hmm_lerner

        self.data_disc_mng = Discretizer(scaled_data_set, N_clusters=features_clusters)

        self.scaled_grouped_data = pd.DataFrame(self.data_disc_mng.predict(self.scaled_data, mode='disc'),
                                                columns=self.scaled_data.columns)

        self.clustered_data = self.data_disc_mng.predict(self.scaled_grouped_data.values, mode='clusters')
        self.scaled_predictions = pd.DataFrame(scaled_data_set[self.pre_column])
        self.predictions_disc_mng = Discretizer(self.scaled_predictions, N_clusters=prediction_clusters, mode='num')
        self.clustered_pred = self.predictions_disc_mng.predict(self.scaled_predictions.values, mode='clusters')

    def test(self):
        # predicted_from_hmm = self.hmm_learner.test(input_data=self.scaled_grouped_data)
        predicted_from_hmm = self.hmm_learner.predict_by_model_scaled
        predicted_disc = self.predictions_disc_mng.predict(predicted_from_hmm, mode='disc')
        return predicted_disc

    def predict_from_clusters(self, X):
        data = []
        for x in X:
            data.append(self.data_disc_mng.clusters_to_centers[x])
        predicted_from_hmm = self.hmm_learner.predict(data)
        state = self.predictions_disc_mng.predict(np.array([predicted_from_hmm]).reshape(1,-1), mode='cluster')
        return state