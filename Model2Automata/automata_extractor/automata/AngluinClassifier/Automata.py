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
            self.classes = {start_class}
            self.Q = ['']
            self.Q_to_classification = {'': start_class}
            self.transitions = {}
            for ltr in self.ALF:
                self.transitions[('', ltr)] = ''

    def add_trans(self, q_s, ltr, q_end):
        self.transitions[(q_s, ltr)] = q_end
        if q_s not in self.Q:
            self.Q.append(q_s)

    def add_new_state(self, s, cls):
        if s not in self.Q:
            self.Q_to_classification[s] = cls
            self.classes.update(cls)
            self.Q.append(s)

    def predict_clusters(self, X):
        return self.predict(X)

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

    def print(self):
        print("Automata")
        print(self.Q)
        print(self.transitions)
        print(self.Q_to_classification)

    def get_shortest_path(self, state, cls):
        explored = []
        queue = [[state]]
        if self.Q_to_classification[state] == cls:
            return 'Cur State'
        while queue:
            path = queue.pop(0)
            last_state = path[-1]
            if last_state not in explored:
                for neighbour in [self.transitions[(last_state, ltr)] for ltr in self.ALF]:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    if self.Q_to_classification[neighbour] == cls:
                        seq = []
                        for s in new_path[1:len(new_path)]:
                            seq.append(list(self.transitions.keys())[list(self.transitions.values()).index(s)][1])
                        return seq
                explored.append(last_state)

        return []

    def reformat_automata(self):
        new_names = []
        for i, q in enumerate(self.Q):
            new_names.append((q, 'Q{}'.format(i)))
        new_Q = []
        new_transitions = {}
        new_Q_to_classifications = {}
        for (old, new) in new_names:
            new_Q.append(new)
            new_Q_to_classifications[new] = self.Q_to_classification[old]
            for ltr in self.ALF:
                for tup in new_names:
                    if tup[0] == self.transitions[(old, ltr)]:
                        new_name = tup[1]
                new_transitions[(new, ltr)] = new_name
        self.Q = new_Q
        self.Q_to_classification = new_Q_to_classifications
        self.transitions = new_transitions

