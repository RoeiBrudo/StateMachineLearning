from ORC import *
from TreeMng import *


class SequenceAndA2AutomataLearner:
    def __init__(self, A, s):
        self.ALF = A.ALF
        self.classes = A.classes
        self.ORC = ORC(A, s)
        self.start = self.ORC.member_query([])
        self.automata = Automata(self.ALF, start_class=self.start)
        is_equiv, counter_example = self.ORC.equiv(self.automata)

        self.init_tree(counter_example)
        self.g = 1
        self.loop()
        print("aa")

    def init_tree(self, counter_example):
        counter_example_class = self.ORC.member_query(counter_example)
        self.tree = NodeTree(self.classes,'')
        self.tree.add_children(self.start, '')
        self.tree.add_children(counter_example_class, str=word2key(counter_example))
        self.tree.add_generic(0)
        
    def loop(self):
        is_equiv = False
        states = -1
        m_hat = None
        new_node = None
        while not is_equiv:
            m_hat = self.get_hyp_from_tree()
            states_1 = len(m_hat.Q_to_classification.keys())
            if states == states_1:
                print("same states num error!!")
            states = states_1
            is_equiv, conter_example = self.ORC.equiv(m_hat)
            if not is_equiv:
                new_node = update_tree(self.ORC, self.tree, word2key(conter_example), m_hat, self.g)
                self.g += 1

        self.automata = m_hat
        print('states', len(m_hat.Q_to_classification.keys()))

    def get_hyp_from_tree(self):
        leafs = list(get_leaves(self.tree))
        automata = Automata(self.ALF, start_class=self.start)

        passed = []
        ready_leafs = set([leaf for leaf in leafs if (not leaf.is_generic and leaf not in passed)])
        while ready_leafs:
            leaf = list(sorted(ready_leafs))[0]
            passed.append(leaf)
            s = leaf.str
            automata.Q_to_classification[s] = self.ORC.member_query(key2word(s))
            for b in self.ALF:
                sb = add_strings(s, b)
                state = sift(self.ORC, self.tree, sb)
                s_t = state.str
                automata.transitions[(s, b)] = s_t
                if state not in passed: ready_leafs.add(state)
            ready_leafs.remove(leaf)

        return automata


if __name__ == '__main__':
    num_alf = 3
    num_states = 4
    num_classes = 2
    for i in range(100):
        A = create_random_automata(num_alf=num_alf, num_classes=num_classes, num_states=num_states)
        s = [random.choice(A.ALF) for i in range(1000)]
        learner = SequenceAndA2AutomataLearner(A, s)
