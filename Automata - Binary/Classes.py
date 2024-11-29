import numpy as np


class Automata:
    def __init__(self, start, Q=['']):
        self.Q = Q
        self.Q_classifications = {'': start}
        self.transitions = {('', '0'): '', ('', '1'): ''}

    def predict(self, x, predict_state=False):
        cur = ''
        for i in x:
            cur = self.transitions[(cur, str(i))]
        if predict_state: return cur
        return self.Q_classifications[cur]

    def print(self):
        print("Automata")
        print(self.Q)
        print(self.transitions)
        print(self.Q_classifications)


class ORC:
    def __init__(self, L, s):
        self.L = L
        self.s = s

    def member_query(self, x):
        return self.L.predict(x)

    def equiv(self, M_hat):
        for i in range(1,len(self.s)+1):
            s_pre = self.s[:i]
            out_l = self.L.predict(s_pre)
            out_M = M_hat.predict(s_pre)
            if out_l != out_M:
                return False, s_pre

        return True, ''


class Language:
    def __init__(self, mod, arr):
        self.mod = mod
        self.in_lang_mods = arr

    def predict(self, x):
        sum = np.sum(np.array(x))
        cond = False
        for mod in self.in_lang_mods:
            cond = cond or bool(int(sum % self.mod == mod))
        return int(cond)
