import numpy as np


class ORC:
    def __init__(self, A, s):
        self.A = A
        self.s = s

    def member_query(self, x):
        a = self.A.predict(x)
        return a

    def equiv(self, M_hat):
        m_state = ''
        for i in range(1,len(self.s)+1):
            s_pre = self.s[:i]
            out_l = self.A.predict(s_pre)
            out_M, m_state = M_hat.read_letter(m_state, self.s[i-1])
            if out_l != out_M:
                return False, s_pre

        return True, ''