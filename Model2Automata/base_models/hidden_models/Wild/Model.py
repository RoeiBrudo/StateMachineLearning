# First, let's define a RNN Cell, as a layer subclass.
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Input, Dense, Activation, RNN
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

opt = keras.optimizers.Adam(learning_rate=0.001)


class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class WildModel:

    def __init__(self):
        self.h = None
        self.m = None
        self.x_to_y_model = None
        self.x_to_hidden_model = None
        self.hidden_to_y_model = None

    def build_model(self, m, h):
        cell = MinimalRNNCell(h)
        x = keras.Input((None, m))
        layer = RNN(cell)
        y = layer(x)

        self.x_to_y_model = Model(x, y)
        self.x_to_y_model.compile(optimizer=opt, loss='mse')

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_to_y_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

    def predict_x_to_y(self, X):
        return self.x_to_y_model.predict(X)

    def predict_x_to_hidden(self, X):
        return self.x_to_hidden_model.predict(X)

    def predict_hidden_to_y(self, X, H):
        return self.hidden_to_y_model.predict([X, H])

    def save_model(self, path):
        self.x_to_y_model.save(path + '_x_to_y')
        self.x_to_hidden_model.save(path + '_x_to_hidden')
        self.hidden_to_y_model.save(path + '_hidden_to_y')

    def load_model(self, path):
        self.h = int(path.split('_')[-2])
        self.M = int(path.split('_')[-1])
        self.x_to_y_model = keras.models.load_model(path + '_x_to_y')
        self.x_to_hidden_model = keras.models.load_model(path + '_x_to_hidden')
        self.hidden_to_y_model = keras.models.load_model(path + '_hidden_to_y')

    def default_h(self):
        hidden = np.zeros(self.h)
        hidden[0] = 1.0
        return hidden

    def default_y(self):
        return 0.5



