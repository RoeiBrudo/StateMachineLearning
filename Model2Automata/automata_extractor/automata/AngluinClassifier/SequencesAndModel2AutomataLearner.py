from automata_extractor.automata.AngluinClassifier import ORC, Automata, helpers, TreeMng


class SequencesAndModel2AutomataLearner:
    def __init__(self, ALF, classes, model, sequences):
        self.ALF = ALF
        self.classes = classes
        self.ORC = ORC.ORC(model, sequences)
        self.start = self.ORC.member_query([])
        self.automata = Automata.Automata(self.ALF, start_class=self.start)
        is_equiv, counter_example = self.ORC.equiv(self.automata)

        self.tree = None
        self.init_tree(counter_example)
        self.g = 1
        self.loop()
        print('Num of automata states = ', len(self.automata.Q_to_classification.keys()))

    def init_tree(self, counter_example):
        counter_example_class = self.ORC.member_query(counter_example)
        self.tree = TreeMng.NodeTree(self.classes, '')
        self.tree.add_children(self.start, '')
        self.tree.add_children(counter_example_class, str=helpers.word2key(counter_example))
        self.tree.add_generic(0)

    def loop(self):
        is_equiv = False
        states = -1
        m_hat = None
        while not is_equiv:
            m_hat = self.get_hyp_from_tree()
            print('states', len(m_hat.Q_to_classification.keys()))
            states_1 = len(m_hat.Q_to_classification.keys())
            if states == states_1:
                print("same states num error!!")
            states = states_1
            is_equiv, conter_example = self.ORC.equiv(m_hat)
            if not is_equiv:
                TreeMng.update_tree(self.ORC, self.tree, helpers.word2key(conter_example), m_hat, self.g)
                self.g += 1

        self.automata = m_hat

    def get_hyp_from_tree(self):
        leafs = list(TreeMng.get_leaves(self.tree))
        automata = Automata.Automata(self.ALF, start_class=self.start)

        passed = []
        ready_leafs = set([leaf for leaf in leafs if (not leaf.is_generic and leaf not in passed)])
        while ready_leafs:
            leaf = list(sorted(ready_leafs))[0]
            passed.append(leaf)
            s = leaf.str
            cls = self.ORC.member_query(helpers.key2word(s))
            automata.add_new_state(s, cls)
            for b in self.ALF:
                sb = helpers.add_strings(s, b)
                state = TreeMng.sift(self.ORC, self.tree, sb)
                s_t = state.str
                automata.add_trans(s, b, s_t)
                if state not in passed:
                    ready_leafs.add(state)
            ready_leafs.remove(leaf)
        return automata