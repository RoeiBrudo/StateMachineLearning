import random
from Automata import *


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

    return Automata(ALF, Q=states, Q_to_classes=Q_to_classes, Q_transitions=transitions)


def word2key(arr):
    if not arr:
        return ''
    else:
        str = ''
        for i in arr:
            if len(str) != 0:
                str = str + ',' + i
            else:
                str = i
        return str


def key2word(str):
    if str == '':
        return []
    else:
        return str.split(',')


def add_strings(a, b):
    l_a = len(a)
    l_b = len(b)
    if l_a == 0 and l_b == 0:
        return ''
    elif l_a == 0:
        return b
    elif l_b == 0:
        return a
    else:
        return a + ',' + b


def get_all_prefix(str):
    prefixes = []
    word = key2word(str)
    for i in range(len(word)):
        str_lst = []
        for j in range(i+1):
            str_lst.append(word[j])
        prefixes.append(word2key(str_lst))
    return prefixes