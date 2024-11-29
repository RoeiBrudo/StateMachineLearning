# from optparse import OptionParser
import os
from hmmr import *
import pandas as pd
from learn import *
from baum_welsh import pre_process_to_hmmr
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
import seaborn as sns


def error_plot(df, features_columns, directory, state):
    residuals = df['diff']
    for x in features_columns:
        x_axis = df[x]
        plt.plot(x_axis, residuals, "b*", label="residuals")
        plt.title('residuals of {}, variable_{}'.format(state, x))
        plt.ylabel('error')
        plt.xlabel(x)
        plt.hlines(y=0, xmin=0, xmax=max(x_axis), colors='r', linestyles='-', lw=2)
        plt.savefig(directory + '/residuals_{}, variable_{}'.format(state, x))
        plt.close()


def plot_viterbi_states(time_index, predicted_states, true_states, directory):
    """
    :param time_index:
    :param predicted_states:
    :param true_states:
    :param directory:
    :return: plot two graphs - 1. states real vs predicted
                               2. confusion materix
    """
    pred_state = [int(i.replace('S', '').replace('s', '')) for i in predicted_states]
    true_state = [int(i.replace('S', '').replace('s', '')) for i in true_states]

    plt.plot(time_index, pred_state, "r^", label="predicted")
    plt.plot(time_index, true_state, "b-", label="true")
    plt.legend()
    plt.title('Viterbi States')
    plt.xlim(0, )
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.savefig(directory + '/viterbi_graph.png')
    plt.close()

    conf_mat = confusion_matrix(pred_state, true_state)
    sns.heatmap(conf_mat, annot=True)
    plt.savefig(directory + '/confusion_mat.png')
    plt.close()


def get_regression_results(hmmr, time_index, in_seq_no_time, y_seq, predicted_states, directory):
    """
    :param hmmr:
    :param time_index:
    :param in_seq_no_time:
    :param y_seq:
    :param predicted_states:
    :param directory:

    plot real vs predicted (axis - time)
    :return df with Time, pred_y, true_y, pred_state
    """
    results_df = pd.DataFrame(in_seq_no_time, copy=True)
    results_df['Time'] = time_index
    pred_y = np.array([0.0] * len(y_seq))
    results_df['true_y'] = list(y_seq)
    results_df['pred_state'] = predicted_states

    # create plot per state
    for state in hmmr.states:
        state_regressor = hmmr.regressions_models[state]
        states_bool = np.where(np.array(predicted_states) == state)[0]
        clean_predict = state_regressor.predict_clean(in_seq_no_time.iloc[states_bool])
        pred_y[states_bool] = clean_predict

    results_df['pred_y'] = pred_y
    results_df['diff'] = y_seq - pred_y

    plt.plot(time_index, results_df['pred_y'], "r^", label="predicted")
    plt.plot(time_index, results_df['true_y'], "b-", label="true")
    plt.legend()
    plt.title('prediction')
    plt.xlim(0, )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(directory + '/results.png')
    plt.close()

    plt.scatter(time_index, results_df['diff'])
    plt.title('residuals')
    plt.xlim(0, )
    plt.savefig(directory + '/residuals.png')
    plt.close()

    return results_df


def regression_id_level(hmmr, in_seq, y_seq, directory):
    """
    do id level testing
    call both plot_viterbi_states and get_regression_results
    :param hmmr:
    :param in_seq:
    :param y_seq:
    :param hidden_states:
    :param directory:
    :return: df with seq results
    """
    time_index = in_seq['Time']
    in_seq_no_time = in_seq.drop(columns=['Time'])
    predicted_states = hmmr.viterbi(in_seq_no_time, y_seq)
    df = get_regression_results(hmmr, time_index, in_seq_no_time, y_seq, predicted_states, directory)
    return df


def test_hmm(hmmr, tagged_data, directory):
    in_sequences, y_sequences = pre_process_to_hmmr(tagged_data, time=True)
    hidden_states = [tagged_data[tagged_data.index == id]['state'].values for id in tagged_data.index.unique()]

    for id, (in_seq, y_seq) in enumerate(zip(in_sequences, y_sequences)):
        id_directory = directory + '/{}'.format(id)
        if not os.path.exists(id_directory):
            os.makedirs(id_directory)

        time_index = in_seq['Time']
        in_seq_no_time = in_seq.drop(columns=['Time'])
        predicted_states = hmmr.viterbi(in_seq_no_time, y_seq)
        plot_viterbi_states(time_index, predicted_states, hidden_states[id], id_directory)


def analyze_hmm(hmmr, unlabeled_data, directory):

    in_sequences, y_sequences = pre_process_to_hmmr(unlabeled_data, time=True)
    features_columns = get_columns_of_data(unlabeled_data, time=True)

    df_all_ids = []
    for id, (in_seq, y_seq) in enumerate(zip(in_sequences, y_sequences)):
        id_directory = directory + '/{}'.format(id)
        if not os.path.exists(id_directory):
            os.makedirs(id_directory)

        df = regression_id_level(hmmr, in_seq, y_seq, id_directory)
        df_all_ids.append(df)

    df_all_ids = pd.concat(df_all_ids)
    for s in hmmr.states:

        state_directory = directory + '/{}'.format(s)
        if not os.path.exists(state_directory):
            os.makedirs(state_directory)

        only_s_df = df_all_ids[df_all_ids.pred_state == s]
        error_plot(only_s_df, features_columns, state_directory, s)