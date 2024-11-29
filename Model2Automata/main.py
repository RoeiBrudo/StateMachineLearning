import config
import numpy as np
import helpers
import argparse
import os

import discrete_model.discrete_model as discrete_model
import automata_extractor.AutomataConverter as AutomataConverter
# from data.Transformation.kbc_transform import Transformation
import automata_wrapper


"""
Module parts:

* Clustering methods - for both x and hidden states / y
* Model - base model with or without hidden states
* 

"""

parser = argparse.ArgumentParser(description="AutomataCli")

data_sets = ['synt_data_rhlp']
base_models_no_hidden = ['rhlp']
base_models_hidden = ['rhlp_hidden']
clustering_methods = ['k-means', 'k-means-rep']

base_model_group = parser.add_argument_group()

# base_model_group.add_argument("--no_hidden_model", dest="no_hidden_model", choices = base_models_no_hidden, default='rhlp')
# base_model_group.add_argument("--hidden_model", dest="hidden_model", choices = base_models_hidden, default=None)

base_model_group.add_argument("--no_hidden_model", dest="no_hidden_model", choices = base_models_no_hidden, default=None)
base_model_group.add_argument("--hidden_model", dest="hidden_model", choices = base_models_hidden, default='rhlp_hidden')

parser.add_argument("--data", dest="data", choices = data_sets, default='synt_data_rhlp')
parser.add_argument("--x_clustering", dest="x_clustering", choices = clustering_methods, default='k-means')
parser.add_argument("--prediction_clustering", dest="prediction_clustering", choices = clustering_methods, default='k-means')


parser.add_argument("--results-folder", dest="results_folder", type=str, default=config.OUT_FOLDER,
                    help="results folder path")

parser.add_argument("--n-for-x", dest="num_of_x_clusters", type=int, default=config.INPUT_CLASSES)
parser.add_argument("--n-for-y", dest="num_of_y_clusters", type=int, default=config.OUTPUT_CLASSES)

args = parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)

    data_func = config.get_data_func(args)
    x_clustering_model, prediction_clustering_model = config.get_clustering_method(args)
    base_model_wrapped = config.get_base_model(args)

    x_train, y_train, x_test, y_test = data_func()

    X_train_2d = np.concatenate(x_train)
    X_test_2d = np.concatenate(x_test)
    Y_train_1d = np.concatenate(y_train)
    Y_test_1d = np.concatenate(y_test)

    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    # transformer = Transformation()
    # X_train, y_train, X_test, y_test = transformer.transform(X_train, y_train, X_test, y_test)

    X_discretizer = x_clustering_model(X_train_2d, args.num_of_x_clusters, mode='ltr')

    if args.hidden_model:
        hidden = True
        pred_for_disc = base_model_wrapped.predict_x_to_hidden(X_train_2d, return_seq=True)
        pred_discretizer = prediction_clustering_model(pred_for_disc, args.num_of_y_clusters, mode='num')

    else:
        hidden = False
        pred_for_disc = np.array(Y_train_1d).reshape((-1, 1))
        pred_discretizer = prediction_clustering_model(pred_for_disc, args.num_of_y_clusters, mode='num')

    discretizer_model = discrete_model.DiscretizedModel(base_model_wrapped, X_discretizer, pred_discretizer)

    # helpers.plot_model_results(base_model_wrapped, x_train, y_train)
    # helpers.plot_disc_model_results(discretizer_model, x_train, y_train)
    # helpers.plot_model_results_single(base_model_wrapped, x_test[0, :, :], y_test[0, :], args.results_folder)
    # helpers.plot_disc_model_results_single(base_model_wrapped, discretizer_model, x_test[0, :, :], y_test[0, :], args.results_folder)

    train_seqs = []

    for i, seq in enumerate(x_train):
        train_seqs.append(X_discretizer.predict_from_vectors(seq, to_vectors=False))

    auto_learner = AutomataConverter.Model2Auto(discretizer_model, train_seqs)
    # learned_automata = auto_learner.write_files(args.results_folder)

    automata_wrap = automata_wrapper.AutomataWrapper(auto_learner.automata, X_discretizer, pred_discretizer)
    helpers.plot_disc_auto_model_results(discretizer_model, automata_wrap, x_train, args.results_folder)
    helpers.plot_disc_auto_model_results_single(discretizer_model, automata_wrap, x_test[0, :, :], y_test[0, :], args.results_folder)
