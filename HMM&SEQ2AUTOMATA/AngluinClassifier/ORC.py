import numpy as np


class ORC:
    def __init__(self, hmm, s):
        self.hmm = hmm
        self.s = s
        self.out_L = [self.member_query(self.s[:i]) for i in range(1, len(self.s)+1)]

    def member_query(self, x):
        # print('start mem', x)
        a = self.hmm.predict_from_clusters(x)[0]
        # print('end mem')
        return a

    def equiv(self, M_hat):
        # print('start equiv')
        m_state = ''
        for i in range(1,len(self.s)+1):
            s_pre = self.s[:i]
            out_l = self.out_L[i-1]
            out_M, m_state = M_hat.read_letter(m_state, self.s[i-1])
            if out_l != out_M:
                # print("counter Example", len(s_pre))
                return False, s_pre
        return True, ''