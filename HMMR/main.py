import os
import pandas as pd

from data_generator import *
from learn import *
from test import *
from explore import *
from hmmr import *


def generate_data(m, max_p, r, n, k, results_folder):
    true_probs, true_transitions, true_emissions, data = sample_model_and_data(m, max_p, r, n, k)

    true_model_folder = results_folder + '/true_model'
    samples_folder = results_folder + '/data'

    if not os.path.exists(true_model_folder):
        os.makedirs(true_model_folder)

    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    true_probs.to_csv(true_model_folder + '/initial_prob.csv', index_label='s')
    true_transitions.to_csv(true_model_folder + '/transitions.csv', index_label='s')
    for state in true_emissions.keys():
        true_emissions[state].to_csv(true_model_folder + '/regression_{}.csv'.format(state), index_label='s')

    data.to_csv(samples_folder + '/data_with_states.csv', index=False)
    no_states_data = data.drop(columns=['state'])
    no_states_data.to_csv(samples_folder + '/data.csv', index=False)


def parse_args():
    from optparse import OptionParser
    usage_str = """
    USAGE:      python game.py <options>
                Insert: -p directory to parameters, 
                -n length of each sample
                -m number of sequences to sample
    """
    parser = OptionParser(usage_str)

    parser.add_option('-f', '--function',
                      dest='function',
                      type='choice',
                      choices=['generate', 'learn', 'analyze', 'test', 'all'],
                      help='main function to execute')

    parser.add_option('-m', dest='m', type='int', help='m')
    parser.add_option('-p', dest='p', type='int', help='max_p')
    parser.add_option('-r', dest='r', type='int', help='r')
    parser.add_option('-n', dest='n', type='int', help='n')
    parser.add_option('-k', dest='k', type='int', help='k')

    parser.add_option('--input_unlabeled',
                      dest='input_unlabeled',
                      help='unlabeled data',
                      default=None)

    parser.add_option('--input_labeled',
                      dest='input_labeled',
                      help='labeled data',
                      default=None)

    parser.add_option('-o', '--output_folder',
                      dest='output_folder',
                      help='destination folder for results',
                      default=None)

    parser.add_option('--regression_type',
                      dest='regression_type',
                      help='regression type',
                      default=None)

    parser.add_option('--start_method',
                      dest='start_method',
                      type='choice',
                      choices=['from_dir', 'from_ranges', 'init_from_data'],
                      help='start method')

    parser.add_option('--number_of_starts',
                      dest='number_of_starts',
                      type='int',
                      help='number of starts points for hmm')

    # from dir
    parser.add_option('--start_dir',
                      dest='start_dir',
                      help='start parameters dir',
                      default=None)

    parser.add_option('--coef_noise',
                      dest='coef_noise',
                      type='float',
                      help='coefficient noise from ',
                      default=None)

    # from ranges

    parser.add_option('--range_of_n_states',
                      dest='range_of_n_states',
                      type='int',
                      nargs=2,
                      help='range of n_states values')

    parser.add_option('--range_of_p',
                      dest='range_of_p',
                      type='int',
                      nargs=2,
                      help='range of p values')

    parser.add_option('--range_of_coeff',
                      dest='range_of_coeff',
                      type='float',
                      nargs=2,
                      help='range of regression coefficients')

    parser.add_option('--transitions_alpha',
                      dest='transitions_alpha',
                      type='float',
                      help='max of alpha in transition matrix')


    args, __ = parser.parse_args()
    return args




if __name__ == '__main__':

    args = parse_args()

    if args.function == 'generate':
        m, max_p, r, n, k = args.m, args.p, args.r, args.n, args.k
        results_folder = args.output_folder
        generate_data(m, max_p, r, n, k, results_folder)

    if args.function == 'learn':
        number_of_starts = args.number_of_starts
        results_folder = args.output_folder
        Xy = pd.read_csv(args.input_unlabeled, index_col='ID')

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        if args.start_method == 'from_dir':
            initial_probs, transitions, regressions = get_params_from_dir(args.start_dir)
            learned_hmm = learn_hmm_based_on_estimators(initial_probs, transitions, regressions, Xy,
                                                        args.number_of_starts, args.coef_noise)
            learned_hmm.write_files(results_folder)

        if args.start_method == 'from_ranges':
            learned_hmm = learn_hmm_random_starts(args.range_of_n_states, args.range_of_p, args.range_of_coeff,
                                                  args.transitions_alpha, args.number_of_starts, Xy)
            learned_hmm.write_files(results_folder)

    if args.function == 'analyze':
        model_folder = args.start_dir
        initial_probs, transitions, regressions = get_params_from_dir(model_folder)
        hmmr = HMMR(initial_probs, transitions, regressions)

        output_analysis_dir = model_folder + '/analysis/'
        if not os.path.exists(output_analysis_dir):
            os.makedirs(output_analysis_dir)

        Xy = pd.read_csv(args.input_unlabeled, index_col='ID')
        analyze_hmm(hmmr, Xy, output_analysis_dir)

    if args.function == 'test':
        model_folder = args.start_dir
        initial_probs, transitions, regressions = get_params_from_dir(model_folder)
        hmmr = HMMR(initial_probs, transitions, regressions)

        output_analysis_dir = model_folder + '/analysis/'
        if not os.path.exists(output_analysis_dir):
            os.makedirs(output_analysis_dir)

        Xy_s = pd.read_csv(args.input_labeled, index_col='ID')
        test_hmm(hmmr, Xy_s, output_analysis_dir)

    if args.function == 'all':
        m, max_p, r, n, k = args.m, args.p, args.r, args.n, args.k
        results_folder = args.output_folder
        generate_data(m, max_p, r, n, k, results_folder)

        Xy = pd.read_csv(results_folder + '/data/data.csv', index_col='ID')
        Xy_s = pd.read_csv(results_folder + '/data/data_with_states.csv', index_col='ID')

        start_model_dir = results_folder + "/true_model"
        model_folder = results_folder + "/learned_model"

        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        if args.start_method == 'from_dir':
            initial_probs, transitions, regressions = get_params_from_dir(start_model_dir)
            learned_hmm = learn_hmm_based_on_estimators(initial_probs, transitions, regressions, Xy,
                                                        args.number_of_starts, args.coef_noise)
            learned_hmm.write_files(model_folder)

        if args.start_method == 'from_ranges':
            learned_hmm = learn_hmm_random_starts(args.range_of_n_states, args.range_of_p, args.range_of_coeff,
                                                  args.transitions_alpha, args.number_of_starts, Xy)
            learned_hmm.write_files(model_folder)

        initial_probs, transitions, regressions = get_params_from_dir(model_folder)
        hmmr = HMMR(initial_probs, transitions, regressions)

        output_analysis_dir = model_folder + '/analysis/'
        if not os.path.exists(output_analysis_dir):
            os.makedirs(output_analysis_dir)

        analyze_hmm(hmmr, Xy, output_analysis_dir)
        test_hmm(hmmr, Xy_s, output_analysis_dir)
