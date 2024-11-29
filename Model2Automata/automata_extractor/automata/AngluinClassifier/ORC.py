class ORC:
    def __init__(self, model, sequences):
        print('start Orc')
        self.model = model
        self.out_L = []
        self.sequences = sequences

        for i, seq in enumerate(sequences):
            self.out_L.append(self.member_query(seq, return_seq=True))

    def member_query(self, x, return_seq=False):
        prediction = self.model.predict(x, from_vectors=False, to_vectors=False, return_seq=return_seq)
        return prediction

    def equiv(self, M_hat):
        for i, s in enumerate(self.sequences):
            m_state = ''
            for j in range(1, len(s)+1):
                s_pre = s[:j]
                out_l = self.out_L[i][j-1]
                out_M, m_state = M_hat.read_letter(m_state, s[j-1])
                if out_l != out_M:
                    print(out_l, out_M)
                    print("counter Example, seq number", i, 'length ', len(s_pre))
                    return False, s_pre
        return True, ''
