import pandas as pd
import automata_extractor.automata.AngluinClassifier.Automata as auto


class AutomataAnalysis:
    def __init__(self, in_dir):
        self.Automata = self.read_automata_from_files(in_dir)

    @staticmethod
    def read_automata_from_files(out_dir):
        transitions_df = pd.read_csv(out_dir + '/Automata_Transitions.csv')
        Q_to_classes_df = pd.read_csv(out_dir + '/Automata_Classifications_group.csv')

        ALF = transitions_df.Cluster.unique()
        transitions = {}
        for i in range(transitions_df.shape[0]):
            transitions[(transitions_df.iloc[i]['Source State'], transitions_df.iloc[i]['Cluster'])] =\
                transitions_df.iloc[i]['Target State']

        Q = []
        q_to_classes = {}
        for i in range(Q_to_classes_df.shape[0]):
            Q.append(Q_to_classes_df.iloc[i]['State'])
            q_to_classes[Q_to_classes_df.iloc[i]['State']] = Q_to_classes_df.iloc[i]['Prediction Group']

        Automata = auto.Automata(ALF, Q=Q, Q_to_classes=q_to_classes, Q_transitions=transitions)
        return Automata

    def find_shortest_transitions(self, start_state, end_cls):
        path = self.Automata.get_shortest_path(start_state, end_cls)
        return path