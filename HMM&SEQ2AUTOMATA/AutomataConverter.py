from AngluinClassifier import SequenceAndHMM2AutomataLearner as Learner
import csv


class Hmm2Auto:
    def __init__(self, hmm_model):
        self.hmm = hmm_model
        self.features_disc_mng = self.hmm.data_disc_mng
        self.pred_disc_mng = self.hmm.predictions_disc_mng
        self.ALF = self.features_disc_mng.clusters_to_centers.keys()
        self.states = self.pred_disc_mng.clusters_to_centers.keys()
        self.automata_learner = Learner.SequenceAndHMM2AutomataLearner(self.ALF, self.states, self.hmm,
                                                                       self.hmm.clustered_data)
        self.correspond_automata = self.automata_learner.automata
        self.get_files()

    def get_files(self):
        with open('Transitions.csv', 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            states = []
            for (i, key) in enumerate(self.correspond_automata.transitions.keys()):
                if key[0] not in states:
                    states.append(key[0])
            for (i, key) in enumerate(self.correspond_automata.transitions.keys()):
                filewriter.writerow(
                    ['Q{}'.format(states.index(key[0])), key[1],
                     'Q{}'.format(states.index(self.correspond_automata.transitions[key]))])
        csvfile.close()

        with open('Q_Classifiations.csv', 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for (i, key) in enumerate(self.correspond_automata.Q_to_classification.keys()):
                filewriter.writerow(
                    ['Q{}'.format(i), self.correspond_automata.Q_to_classification[key]])
        csvfile.close()
#