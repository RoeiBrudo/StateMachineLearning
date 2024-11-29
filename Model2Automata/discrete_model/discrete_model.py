import copy
import numpy as np


class DiscretizedModel:
    def __init__(self, base_model, data_discretizer, predictions_discretizer):
        self.base_model = base_model
        self.is_hidden = base_model.is_hidden
        self.X_discretizer = data_discretizer
        self.predictions_discretizer = predictions_discretizer

    def predict(self, X, from_vectors, to_vectors, return_seq=True):
        X = np.array(X)
        if X.size == 0:
            if not to_vectors:
                return '-1'
            else:
                print("Something Is Wrong")

            representative_X = X
        else:
            if from_vectors:
                representative_X = self.X_discretizer.predict_from_vectors(X, to_vectors=True)
            else:
                representative_X = self.X_discretizer.get_clusters_representatives(X)

        if self.is_hidden:
            predicted_from_model = self.base_model.predict_x_to_hidden(representative_X)
        else:
            predicted_from_model = self.base_model.predict_x_to_y(representative_X)

        if to_vectors:
            predicted_disc = self.predictions_discretizer.predict_from_vectors(predicted_from_model, to_vectors=True)
        else:
            predicted_disc = self.predictions_discretizer.predict_from_vectors(predicted_from_model, to_vectors=False)

        if return_seq:
            return predicted_disc
        else:
            return predicted_disc[-1]

    def predict_x_to_y_with_clustering(self, X, return_seq = True):
        if not self.is_hidden:
            return self.predict(X, from_vectors=True, to_vectors=True)
        else:
            representative_X = self.X_discretizer.predict_from_vectors(X, to_vectors=True)
            predicted_disc = self.predict(X, from_vectors=True, to_vectors=True)
            predicted_y = self.base_model.predict_hidden_to_y(representative_X, predicted_disc)

        next_prediction = copy.copy(predicted_y)
        if not return_seq:
            return next_prediction[-1]
        else:
            return next_prediction
