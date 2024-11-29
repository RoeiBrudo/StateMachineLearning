import numpy as np
from base_models.hidden_model_abs import HiddenModelAbstract

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

opt = keras.optimizers.Adam(learning_rate=0.001)


class RHLPModel(HiddenModelAbstract):

    def __init__(self):
        self.is_hidden = True
        self.x_to_y_model = None
        self.x_to_hidden_model = None
        self.hidden_to_y_model = None

    def build_model(self, m, h, beta):
        """
        :param m: input features dimention
        :param h: hidden state vector length
        :param beta: beta coefficient of softmax activation function
        :return:
        """
        softmax_func = self.modified_softmax(beta)

        input_x = Input(shape=(m,))
        h_logistic_regression = Dense(h, use_bias=True)(input_x)
        hidden_vector = Activation(softmax_func)(h_logistic_regression)
        regressions = Dense(h, use_bias=True)(input_x)
        y_matrix = K.dot(hidden_vector, K.transpose(regressions))
        y = tf.linalg.diag_part(y_matrix)

        hidden_states_from_input = Input(shape=(h,))
        y_matrix_from_input_calc = K.dot(hidden_states_from_input, K.transpose(regressions))
        y_from_input_calc = tf.linalg.diag_part(y_matrix_from_input_calc)

        self.x_to_y_model = Model(input_x, y)
        self.x_to_y_model.compile(optimizer=opt, loss='mse')
        self.x_to_hidden_model = Model(input_x, hidden_vector)
        self.hidden_to_y_model = Model([input_x, hidden_states_from_input], y_from_input_calc)

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_to_y_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

    def predict_x_to_y(self, X, return_seq=True):

        prediction = self.x_to_y_model(X)
        if not return_seq:
            return prediction[-1]
        else:
            return prediction

    def predict_x_to_hidden(self, X, return_seq=True):

        prediction = np.array(self.x_to_hidden_model(X), dtype=np.float)
        if return_seq:
            if not return_seq:
                return prediction[-1]
            else:
                return prediction

    def predict_hidden_to_y(self, X, H, return_seq=True):

        prediction = self.hidden_to_y_model([X, H])
        if not return_seq:
            return prediction[-1]
        else:
            return prediction

    def save_model(self, path):
        self.x_to_y_model.save(path + '_x_to_y')
        self.x_to_hidden_model.save(path + '_x_to_hidden')
        self.hidden_to_y_model.save(path + '_hidden_to_y')

    def load_model(self, path):
        self.x_to_y_model = keras.models.load_model(path + '_x_to_y')
        self.x_to_hidden_model = keras.models.load_model(path + '_x_to_hidden')
        self.hidden_to_y_model = keras.models.load_model(path + '_hidden_to_y')

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
