from HMMDisc import *
import matplotlib.pyplot as plt
from DataExtractor import *
from AutomataConverter import *
from mpl_toolkits.mplot3d import axes3d

set_num = 3

file_name = 'Data/occupancy_data/datatest.csv'
data_columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']
predict_column = 'Occupancy'
data_ext = DataExtractor()
scaled_data_set = data_ext.load_and_scale_data_set(file_name, data_columns, predict_column, mean_over=2)
predict_index = list(scaled_data_set.columns).index(predict_column)


def get_automata(days_to_test, N_inner_states,
                 N_predictions_clusters, N_features_clusters):

    hmm_learner = HmmLerner(scaled_data_set, predict_column, days_to_test, N_inner_states)
    hmm_predictions_scaled = hmm_learner.test()
    hmm_predictions = data_ext.scale_back(predict_index, hmm_predictions_scaled)

    explanatory_learner = HmmDisc(scaled_data_set, predict_column,
                                  features_clusters=N_features_clusters, prediction_clusters=N_predictions_clusters,
                                  hmm_lerner=hmm_learner)
    exp_scaled_predictions = explanatory_learner.test()
    exp_hmm_predictions = data_ext.scale_back(predict_index, exp_scaled_predictions)

    actual_results = data_ext.scale_back(predict_index, scaled_data_set.iloc
                                        [scaled_data_set.shape[0] - hmm_learner.N_test:scaled_data_set.shape[0],
                                        predict_index: predict_index + 1])
    actual_data = data_ext.scale_back(predict_index, scaled_data_set.iloc
                                        [: ,predict_index : predict_index + 1])


    fig = plt.figure()
    days = np.arange(hmm_learner.N_test)
    whole_days = np.arange(actual_data.shape[0])
    axes = fig.add_subplot(111)

    plt.title("InnerStates={}, featuresClusters={}, prediction_cluster={}".format(
                                    N_inner_states, N_features_clusters, N_predictions_clusters))

    import os

    main = 'img/s{}'.format(set_num)

    results_dir = os.path.join(main, 'Inner{}/'.format(N_inner_states))

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    axes.plot(days, actual_results, 'bo-', label="actual")
    axes.plot(days, hmm_predictions, 'r+-', label="hmm predicted")
    axes.plot(days, exp_hmm_predictions, 'p-', label="explanatory predicted")
    # axes.plot(whole_days, actual_data, 'p-', label="explanatory predicted")
    plt.legend()
    plt.show()
    plt.savefig(results_dir + "Pred={},Feat={}".format(N_predictions_clusters,N_features_clusters))
    plt.close()

    get_files(explanatory_learner)

    auto_learner = Hmm2Auto(explanatory_learner)
    return auto_learner.correspond_automata


def get_files(explanatory_learner):
    with open('Features_Clusters.csv', 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for c in explanatory_learner.data_disc_mng.clusters_to_centers:
            filewriter.writerow([c])
            for key in explanatory_learner.data_disc_mng.properties[c]:
                ind = list(scaled_data_set.columns).index(key)
                filewriter.writerow(['Min {}'.format(key), 'Max {}'.format(key)])
                prop = explanatory_learner.data_disc_mng.properties[c][key]
                filewriter.writerow(data_ext.scale_back(ind, np.array(prop).reshape(len(prop),1)))

    csvfile.close()

    with open('Predictions_Clusters.csv', 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for c in explanatory_learner.predictions_disc_mng.clusters_to_centers:
            filewriter.writerow([c])
            for key in explanatory_learner.predictions_disc_mng.properties[c]:
                filewriter.writerow(['Min {}'.format(key), 'Max {}'.format(key)])
                ind = list(scaled_data_set.columns).index(key)
                prop = explanatory_learner.predictions_disc_mng.properties[c][key]
                filewriter.writerow(data_ext.scale_back(ind, np.array(prop).reshape(len(prop),1)))

    csvfile.close()


def get_curve(s, f_u, f_d, p_u, p_d, q):
    x = [range(f_d, f_u, 2)]
    y = [range(p_d,p_u, 2)]
    X, Y = np.meshgrid(x, y)
    Z = np.array(q).reshape(X.shape[0], X.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("N Hidden states = {}".format(s))
    ax.set_xlabel("N features classes")
    ax.set_xticks(np.arange(f_d, f_u, step=1))
    ax.set_ylabel("N prediction classes")
    ax.set_yticks(np.arange(p_d, p_u, step=1))
    ax.set_zlabel("N automata states")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

    plt.savefig("img/s1/curve_for_{}_states".format(s))
    plt.close()




if __name__ == "__main__":
    # f_d, f_u = 10, 25
    # p_d, p_u = 10, 25
    # for n_h_s in range(20, 42, 4):
    #     q = []
    #     for f in range(f_d, f_u, 2):
    #         for p in range(p_d, p_u, 2):
    #             print('hidden states = {}   {} features and {} predediction classes'.format(n_h_s, f, p))
    #             automata = get_automata(15, n_h_s, p, f)
    #             s = len(automata.Q_to_classification.keys())
    #             q.append(s)

        # get_curve(n_h_s, f_u, f_d, p_u, p_d, q)
    # param 1 - test days
    # param 2 - number of hidden states hmm
    # param 3 - prediction clusters
    # param 4 - features clusters

    automata = get_automata(50, 50, 2, 50)
