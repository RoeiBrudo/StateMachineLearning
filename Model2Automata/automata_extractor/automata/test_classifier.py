from automata_extractor.automata.AngluinClassifier import Automata
import random
from automata_extractor.automata.AngluinClassifier.SequencesAndModel2AutomataLearner import SequencesAndModel2AutomataLearner


def create_random_automata(num_alf, num_states, num_classes):
    states = ['Q{}'.format(i) for i in range(num_states)]
    ALF = [str(i) for i in range(num_alf)]
    classes = [chr(65 + i) for i in range(num_classes)]
    transitions = {}
    Q_to_classes = {}

    for q in states:
        Q_to_classes[q] = random.choice(classes)
        for ltr in ALF:
            transitions[(q, ltr)] = random.choice(states)

    return Automata.Automata(ALF, Q=states, Q_to_classes=Q_to_classes, Q_transitions=transitions)


if __name__ == '__main__':
    samples = 100
    l_of_each_sample = 150

    random_automata = create_random_automata(25, 500, 10)
    samples = [[random.choice(random_automata.ALF) for i in range(l_of_each_sample)] for j in range(samples)]

    learned = SequencesAndModel2AutomataLearner(random_automata.ALF,
                                                random_automata.classes,
                                                random_automata, samples)

    # ALF = ['0', '1']
    # states = ['Q0', 'Q1', 'Q2', 'Q3']
    # Q_to_classes = {'Q0':'0',
    #                 'Q1':'0',
    #                 'Q2':'0',
    #                 'Q3':'1'}
    # transitions = {('Q0', '0'):'Q0',
    #                 ('Q0', '1'):'Q1',
    #                 ('Q1', '0'):'Q1',
    #                 ('Q1', '1'):'Q2',
    #                 ('Q2', '0'): 'Q2',
    #                 ('Q2', '1'): 'Q3',
    #                 ('Q3', '0'): 'Q3',
    #                 ('Q3', '1'): 'Q0'
    #                }
    #
    # A =  Automata.Automata(ALF, Q=states, Q_to_classes=Q_to_classes, Q_transitions=transitions)
    # learned = SequencesAndModel2AutomataLearner(A.ALF,
    #                                             A.classes,
    #                                             A, [['1', '1', '0', '1']])
