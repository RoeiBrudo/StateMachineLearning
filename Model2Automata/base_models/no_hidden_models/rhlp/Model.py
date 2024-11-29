import numpy as np
from base_models.no_hidden_model_abs import NoHiddenModelAbstract

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

opt = keras.optimizers.Adam(learning_rate=0.001)


class RHLPModel(NoHiddenModelAbstract):

    def __init__(self):
        super(RHLPModel, self).__init__()
        self.x_to_y_model = None

    def build_model(self, m, h, beta):

        softmax_func = self.modified_softmax(beta)

        input_x = Input(shape=(m,))
        h_logistic_regression = Dense(h, use_bias=True)(input_x)
        hidden_vector = Activation(softmax_func)(h_logistic_regression)

        regressions = Dense(h, use_bias=True)(input_x)
        y_matrix = K.dot(hidden_vector, K.transpose(regressions))
        y = tf.linalg.diag_part(y_matrix)

        self.x_to_y_model = Model(input_x, y)
        self.x_to_y_model.compile(optimizer=opt, loss=keras.losses.mean_squared_error)

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_to_y_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    def predict_x_to_y(self, X, return_seq = True):

        prediction = self.x_to_y_model.predict(X)
        if not return_seq:
            return prediction[-1]
        else:
            return prediction

    def save_model(self, path):
        self.x_to_y_model.save(path)

    def load_model(self, path):
        self.x_to_y_model = keras.models.load_model(path)

    @staticmethod
    def modified_softmax(beta):

        def variable_softmax(z):
            if len(z.shape) != 2:
                raise RuntimeError("Data has too many dimensions")
            s = K.max(z, axis=1)
            s = s[:, np.newaxis]  # necessary step to do broadcasting
            e_x = K.exp(beta * (z - s))
            div = K.sum(e_x, axis=1)
            div = div[:, np.newaxis]  # dito
            return e_x / div

        return variable_softmax
