from base_models.hidden_models.rhlp_hidden.Model import RHLPModel

import numpy as np


class RHLPWrapper:
    def __init__(self):
        self.is_hidden = True

    def load_model(self):
        self.model = RHLPModel()
        self.model.load_model('base_models/hidden_models/rhlp_hidden/rhlp_4_4')

    def predict_x_to_y(self, X, return_seq=True):
        if X.size == 0:
            return np.array([self.model.default_y()])

        prediction = self.model.predict_x_to_y(X)
        if not return_seq:
            return prediction[-1]
        else:
            return prediction

    def predict_x_to_hidden(self, X, return_seq=True):
        if X.size == 0:
            return np.array([self.model.default_h()], dtype=np.float)

        prediction = np.array(self.model.predict_x_to_hidden(X), dtype=np.float)
        if return_seq:
            if not return_seq:
                return prediction[-1]
            else:
                return prediction

    def predict_hidden_to_y(self, X, H, return_seq=True):
        if X.size == 0:
            return 0.5

        prediction = self.model.predict_hidden_to_y(X, H)
        if not return_seq:
            return prediction[-1]
        else:
            return prediction
