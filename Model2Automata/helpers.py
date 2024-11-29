import matplotlib.pyplot as plt
import numpy as np


def plot_model_results_single(model, X_test, y_test, to_file=None, to_show=True):

    X_test_axis = np.arange(X_test.shape[0])
    y_hat_test = []

    for i in range(X_test.shape[0]):
        if i == 0:
            data_t = np.array(X_test[:i+1, :]).reshape((1, -1))
        else:
            data_t = np.array(X_test[:i+1, :])

        y_hat_test.append(model.predict_x_to_y(data_t, return_seq=False))

    plt.plot(X_test_axis, y_hat_test, c='b')
    plt.plot(X_test_axis, y_test, c='r')

    if to_file:
        plt.savefig(to_file + '/base.png')

    if to_show:
        plt.show()
    plt.close()


def plot_disc_model_results_single(base_model, disc_model, X_test, y_test, to_file=None, to_show=True):

    X_test_axis = np.arange(X_test.shape[0])
    y_hat_test_base = base_model.predict_x_to_y(X_test)

    y_hat_test_disc = disc_model.predict_x_to_y_with_clustering(X_test)

    plt.plot(X_test_axis, y_test, c='r')
    plt.plot(X_test_axis, y_hat_test_base, c='b')
    plt.plot(X_test_axis, y_hat_test_disc, c='g', linewidth=3.5)

    if to_file:
        plt.savefig(to_file + '/disc.png')
    if to_show:
        plt.show()
    plt.close()


def plot_disc_auto_model_results_single(disc_model, automata, X_test, y_test, to_file=None, to_show=True):
    X_test_axis = np.arange(X_test.shape[0])
    X_rep = disc_model.X_discretizer.predict_from_vectors(X_test, to_vectors=True)
    y_hat_test_disc = disc_model.predict_x_to_y_with_clustering(X_test, return_seq=True)
    y_auto_test_hidden = []

    for i in range(X_test.shape[0]):
        data_t = np.array(X_test[:i+1, :])
        y_auto_test_hidden.append(automata.predict(data_t))

    if disc_model.is_hidden:
        y_auto_test_hidden = np.array(y_auto_test_hidden)
        y_auto_test_hidden = y_auto_test_hidden.reshape((X_test.shape[0], y_auto_test_hidden.shape[-1]))
        y_auto_test = disc_model.base_model.predict_hidden_to_y(X_rep, y_auto_test_hidden)
    else:
        y_auto_test = np.array(y_auto_test_hidden).flatten()

    plt.plot(X_test_axis, y_test, c='r')
    plt.plot(X_test_axis, y_hat_test_disc, c='g', linewidth=3.5)
    plt.plot(X_test_axis, y_auto_test, c='black', linewidth=2)

    if to_file:
        plt.savefig(to_file + '/auto.png')

    if to_show:
        plt.show()
    plt.close()


## plot 3d data

def plot_model_results(model, X_test, y_test, to_file=None, to_show=True):
    if len(X_test.shape) == 3:
        pred = []
        true_y = []
        for i in range(X_test.shape[0]):
            pred.extend(model.predict_x_to_y(X_test[i, :, :]))
            true_y.extend(y_test[i, :])
    else:
        pred = model.predict_x_to_y(X_test)
        true_y =  y_test

    plt.scatter(true_y, pred, s=0.8)
    plt.show()

    if to_file:
        plt.savefig(to_file + '/base_all.png')

    if to_show:
        plt.show()
    plt.close()


def plot_disc_model_results(model, X_test, y_test, to_file=None, to_show=True):
    if len(X_test.shape) == 3:
        pred = []
        true_y = []
        for i in range(X_test.shape[0]):
            pred.extend(model.predict_x_to_y_with_clustering(X_test[i, :, :]))
            true_y.extend(y_test[i, :])
    else:
        pred = model.predict_x_to_y_with_clustering(X_test)
        true_y = y_test

    plt.scatter(true_y, pred, s=0.8)
    plt.show()

    if to_file:
        plt.savefig(to_file + '/base_all.png')

    if to_show:
        plt.show()
    plt.close()


def auto_to_y(disc_model, automata, X_test):
    X_rep = disc_model.X_discretizer.predict_from_vectors(X_test, to_vectors=True)

    y_auto_test_hidden = []

    for i in range(X_test.shape[0]):
        data_t = np.array(X_test[:i + 1, :])
        y_auto_test_hidden.append(automata.predict(data_t))

    if disc_model.is_hidden:
        y_auto_test_hidden = np.array(y_auto_test_hidden)
        y_auto_test_hidden = y_auto_test_hidden.reshape((X_test.shape[0], y_auto_test_hidden.shape[-1]))
        y_auto_test = disc_model.base_model.predict_hidden_to_y(X_rep, y_auto_test_hidden)
    else:
        y_auto_test = y_auto_test_hidden

    return y_auto_test


def plot_disc_auto_model_results(disc_model, automata, X_test, to_file=None, to_show=True):

    if len(X_test.shape) == 3:
        y_auto_test = []
        y_hat_test_disc = []
        for i in range(X_test.shape[0]):
            y_hat_test_disc.extend(disc_model.predict_x_to_y_with_clustering(X_test[i, :, :]))
            y_auto_test.extend(auto_to_y(disc_model, automata, X_test[i, :, :]))
    else:
        y_hat_test_disc = disc_model.predict_x_to_y_with_clustering(X_test)
        y_auto_test = auto_to_y(disc_model, automata, X_test)

    plt.scatter(y_hat_test_disc, y_auto_test, s=0.8)

    if to_file:
        plt.savefig(to_file + '/auto.png')

    if to_show:
        plt.show()

    plt.close()
