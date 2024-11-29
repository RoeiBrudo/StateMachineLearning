import numpy as np
import csv


class Automata:
    def __init__(self, ALF, Q=None, start_class=None, Q_to_classes=None, Q_transitions=None):
        self.ALF = ALF
        if Q is not None:
            self.Q = Q
            self.Q_to_classification = Q_to_classes
            self.classes = set(self.Q_to_classification.values())
            self.transitions = Q_transitions
        else:
            self.classes = [start_class]
            self.Q = ['']
            self.Q_to_classification = {'': start_class}
            self.transitions = {}
            for ltr in self.ALF:
                self.transitions[('', ltr)] = ''
                
    def add_trans(self, q_s, ltr, q_end):
        self.transitions[(q_s, ltr)] = q_end
        if q_s not in self.Q:
            self.Q.append(q_s)    
        
    def predict(self, x, predict_state=False):
        cur = self.Q[0]
        for i in x:
            cur = self.transitions[(cur, i)]
        if predict_state:
            return cur
        return self.Q_to_classification[cur]

    def read_letter(self, state, letter):
        if state == 'start':
            state = ''
        new_state = self.transitions[(state, letter)]
        return self.Q_to_classification[new_state], new_state

    def get_files(self):
        with open('Transitions.csv', 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            states = []
            for (i, key) in enumerate(self.transitions.keys()):
                if key[0] not in states:
                    states.append(key[0])
            for (i, key) in enumerate(self.transitions.keys()):
                filewriter.writerow(
                    ['Q{}'.format(states.index(key[0])), key[1], states.index(self.transitions[key])])
        csvfile.close()

        with open('Q_Classifiations.csv', 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for (i, key) in enumerate(self.Q_to_classification.keys()):
                filewriter.writerow(
                    ['Q{}'.format(i), self.Q_to_classification[key]])
        csvfile.close()

    def print(self):
        print("Automata")
        print(set([key[0] for key in self.transitions.keys()]))
        print(self.transitions)
        print(self.Q_to_classification)


