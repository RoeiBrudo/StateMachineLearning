from linear_data import *
from multi_lerner import MultiLearner
from classic_learner import ClassicLearners
import matplotlib.pyplot as plt
from helpers import reduce_data
from copy import deepcopy
from keras.models import clone_model

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.keras.backend.set_floatx('float64')


k = 5
d = 80


target_acc = 0.85
# s = 10

max_size = 10
jump = 50
true_classifiers = generate_true_random_classifiers(d, k)
epochs = 500

multi_scores = []
singles_scores = []

train_df = create_data(d, k, true_classifiers, 10000)
test_df = create_data(d, k, true_classifiers, 1000)


def main():
    for iter in range(1, max_size):

        data_size = iter*jump
        train_df_i = train_df.iloc[:data_size].copy()

        print('data size:', data_size)

        multi_model = MultiLearner(d, k)
        train_loss_multi, test_scores_multi = multi_model.fit(train_df_i, test_df, epochs)

        multi_scores.append(test_scores_multi[-1])

        classic_learner = ClassicLearners(d, k)
        train_loss_singles, test_scores_singles = classic_learner.fit(train_df_i, test_df, epochs)
        singles_scores.append(test_scores_singles[-1])

        print('multi learner, test error: ',  test_scores_multi[-1],
              'classical learners, test error:', test_scores_singles[-1])

    plt.plot(list(np.arange(1, max_size)*jump), multi_scores, label='multi learner score')
    plt.plot(list(np.arange(1, max_size)*jump), singles_scores, label='separate learners score')
    plt.title('Scores VS total data size')
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")

    plt.show()


def main2():
    model = MultiLearner(d, k)

    n_ranges = [range(10, 500, 40),
                range(10, 200, 20),
                range(10, 100, 10)] + [range(10, 1000, 10)]*(k-3)

    samples = []
    for k_ in range(1, k+1):
        test_scores_multi = [0]
        i = 0
        final_n_k = -1
        test_data = test_df[test_df.k <= k_]
        while test_scores_multi[-1] < target_acc:
            train_data = reduce_data(train_df, k_, n_ranges[k_-1][i])
            train_loss_multi, test_scores_multi = model.fit(train_data, test_data, epochs)
            print("k: ", k_, "samples", n_ranges[k_-1][i])
            print('score', test_scores_multi[-1])
            multi_scores.append(test_scores_multi[-1])
            i += 1
            if test_scores_multi[-1] > target_acc:
                new_model = MultiLearner(d, k)
                new_model.clone_model(model.model)
                model = new_model
                final_n_k = n_ranges[k_-1][i]

        samples.append(final_n_k)

    print(samples)


if __name__ == '__main__':
    # main2()
    plt.plot(list(np.arange(1, 9)), [210, 110, 70, 44, 26, 42, 22, 22])
    plt.title('# samples to reach accuracy of 0.9')
    plt.legend()
    plt.xlabel("number of classifiers")
    plt.ylabel("Number of samples")

    plt.show()
