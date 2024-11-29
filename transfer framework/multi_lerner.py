import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input


from helpers import split_to_batches_full
import os


class Bias(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Bias, self).__init__(autocast=False)
        self.units = units

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs, **kwargs):
        return inputs + self.b

    def get_config(self):
        return {'units': self.units}


class MultiLearner:

    def __init__(self, d, k):
        self.model = self.__build_model(d, k)
        self.optimizer = optimizers.Adam()

    def clone_model(self, model):
        self.model = clone_model(model)

    def __build_model(self, d, k):
        X = Input(shape=(d))
        l1 = Dense(units=1, use_bias=False)(X)
        y_hat = Bias(units=k)(l1)

        model = Model(inputs=X, outputs=y_hat)
        return model

    # @tf.function
    def __loss_func(self, y_hats, y_trues, ks):
        """
        y_hats - n X k matrix
        y_trues - (n, ) array
        k - (n, ) array
        """
        n, k = y_hats.shape[0], y_hats.shape[1]
        K = np.zeros((n, k))
        K[list(np.arange(n)), ks] = 1
        y_hats_sparse = tf.math.multiply(y_hats, K)
        y_hats_array = tf.reduce_sum(y_hats_sparse, 1)
        return tf.reduce_sum(losses.hinge(y_hats_array, y_trues))

    # @tf.function
    def __train_step(self, x, y, k):
        """
        y_hats - n X k matrix
        y_trues - (n, ) array
        k - (n, ) array
        """
        with tf.GradientTape() as tape:
            y_hat = self.model(inputs=x, training=True)
            loss_value = self.__loss_func(y_hat, y, k)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return loss_value

    # @tf.function
    def score(self, y_hats, y_trues, ks):
        """
        y_hats - n X k matrix
        y_trues - (n, ) array
        k - (n, ) array
        """
        n, k = y_hats.shape[0], y_hats.shape[1]
        K = np.zeros((n, k))
        K[list(np.arange(n)), ks] = 1
        y_hats_sparse = np.multiply(y_hats, K)
        y_hats_array = np.sum(y_hats_sparse, 1)
        res = y_hats_array * y_trues
        res[res <= 0] = 0
        res[res > 0] = 1
        return np.mean(res)

    def fit(self, train_df, test_df, epochs):

        X_train_batches, Y_train_batches, K_train_batches = split_to_batches_full(train_df)
        X_test_batches, Y_test_batches, K_test_batches = split_to_batches_full(test_df)

        train_loss = []
        test_scores = []

        for epoch in range(epochs):
            epoch_train_loss = 0
            for x, y, k in zip(X_train_batches, Y_train_batches, K_train_batches):
                epoch_train_loss += self.__train_step(x, y, k)
            train_loss.append(epoch_train_loss.numpy())

            scores = []
            for x, y, k in zip(X_test_batches, Y_test_batches, K_test_batches):
                y_hats = self.model(inputs=x, training=False)
                scores.append(self.score(y_hats, y, k))
            test_scores.append(np.mean(np.array(scores)))
        return train_loss, test_scores
