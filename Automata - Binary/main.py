from Classes import *
from TreeMng import *
from helpers import *

L = Language(5, [2, 3, 0])

s = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]


class AutomataLearner:
    def __init__(self):
        self.ORC = ORC(L, s)
        start = self.ORC.member_query([])
        self.automata = Automata(start)
        is_equiv, counter_example = self.ORC.equiv(self.automata)
        self.start = self.automata.predict(counter_example)
        if self.start:
            self.tree = NodeTree(str='')
            self.tree.add_children()
            self.tree.add_children(str=arr2str(counter_example))
        else:
            self.tree = NodeTree(str='')
            self.tree.add_children(str=arr2str(counter_example))
            self.tree.add_children()

    def loop(self):
        is_equiv = False
        while not is_equiv:
            m_hat = self.get_hyp_from_tree()
            is_equiv, conter_example =  self.ORC.equiv(m_hat)

            if not is_equiv:
                update_tree(self.ORC, self.tree, conter_example, m_hat)
                # self.tree.print()
                # print("End of Tree")

        m_hat.print()
        return m_hat

    def get_hyp_from_tree(self):
        leafs = list(get_leaves(self.tree))
        leafs_str = [leaf.str for leaf in leafs]
        automata = Automata(start=self.start, Q=leafs_str)
        for leaf in leafs:
            s = leaf.str
            for b in ['0', '1']:
                sb = s+b
                s_t = sift(self.ORC, self.tree, sb)
                automata.transitions[(s, b)] = s_t

            automata.Q_classifications[s] = bool(self.ORC.member_query(str2arr(s)))

        return automata


if __name__ == '__main__':
    learner = AutomataLearner()
    learner.loop()