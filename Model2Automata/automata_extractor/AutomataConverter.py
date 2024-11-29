from automata_extractor.automata.AngluinClassifier import SequencesAndModel2AutomataLearner as Learner
import pandas as pd
import os


class Model2Auto:
    def __init__(self, model, X_disc):
        self.model = model
        self.ALF = self.model.X_discretizer.clusters_to_centers.keys()
        self.classes = self.model.predictions_discretizer.clusters_to_centers.keys()

        self.automata_learner = Learner.SequencesAndModel2AutomataLearner(self.ALF, self.classes, self.model, X_disc)
        self.automata = self.automata_learner.automata
        self.automata.reformat_automata()

    def write_files(self, out_path):
        transitions_df = pd.DataFrame(columns=['Source State', 'Cluster', 'Target State'])
        predictions_df = pd.DataFrame(columns=['State', 'Prediction Group'])
        i = 0
        normalized_states = list(self.automata.Q)

        for (j, state) in enumerate(normalized_states):

            predictions_df.loc[j, 'State'] = state
            predictions_df.loc[j, 'Prediction Group'] = self.automata.Q_to_classification[state]

            for ltr in self.automata.ALF:
                transitions_df.loc[i, 'Source State'] = state
                transitions_df.loc[i, 'Cluster'] = ltr
                transitions_df.loc[i, 'Target State'] = self.automata.transitions[(state, ltr)]

                i += 1

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        transitions_df.to_csv(out_path + '/' + 'Automata_Transitions.csv', index=False)
        predictions_df.to_csv(out_path + '/' + 'Automata_Classifications_group.csv', index=False)

