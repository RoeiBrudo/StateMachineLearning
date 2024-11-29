import numpy as np
import math

# alphabet = ['a', 'b', 'c', 'd', 'e']
# target_u = ['b', 'c', 'd', 'e']
# n = 10


alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
target_u = ['c', 'd', 'b', 'd', 'a', 'a', 'c']
n = 10

s = len(alphabet)
l = len(target_u)
eps = 0.1
delta = 0.1


def nCr(n,k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))


class ShuffleLearner():
    def __init__(self):

        self.p = (nCr(n-1, l-1)) * (1/s)**(l-1)*(1 - 1/s)**(n-l)
        self.m = (((s**2)*(n**2)) / (eps**2)) * (n * (np.log(s) - np.log(delta)))
        self.m = int(self.m)

        self.data_manager = DataManager(self.m)
        self.data_manager.create_data()

        self.learned_u = []

    def learn_u(self):
        while len(self.learned_u) < l:
            print("Learn {}'th symbol".format(len(self.learned_u)+1))
            self.learn_single()
        return self.learned_u

    def learn_single(self):
        for symbol in alphabet:
            print('Testing {} symbol'.format(symbol))
            EX = self.data_manager.statistical_query_x(self.learned_u, symbol)

            if (EX > (2/s)*self.p - 2*eps/(9*(s-1)*n)) and EX < (2/s)*self.p + 2*eps/(9*(s-1)*n):
                self.learned_u.append(symbol)
                print('Learned! Current u = {}'.format(self.learned_u))
                return
            else:
                print("Didn't match!")


class DataManager:
    def __init__(self, m):
        self.m = m
        self.data = []
        self.labels = []

    def create_data(self):
        data = []
        labels = []
        print("Generating {} samples".format(self.m))
        percentage = 10
        parts = np.arange(1,percentage+1)*np.floor(self.m/percentage)
        for i in range(self.m):
            sample = np.random.choice(alphabet, size=n)
            sample = np.ndarray.tolist(sample)
            data.append(sample)
            labels.append(self.query_helper(target_u, sample, 0)[0])
            if i in parts: print('Generated {}% of samples'.format(
                (np.where(parts == i)[0][0]+1)*percentage))

        print("Done Generating!")
        self.data = data
        self.labels = labels

    # recursive algorithm for u_to check is substring of a sample
    # 2.2 in pdf
    def query_helper(self, u_to_check, sample, step):
        if step == len(u_to_check):
            if len(sample) == 0:
                return [1, 'E']
            else:
                return [1, sample[0]]

        if target_u[step] in sample:
            j = sample.index(target_u[step])
            return self.query_helper(u_to_check, sample[j + 1:len(sample)], step + 1)
        else:
            return [-1, 'E']

    def statistical_query_x(self, u_prime, symbol):
        sum_of_x = 0
        for (x, y) in zip(self.data, self.labels):
            x_prime = x[0:len(x)-1]
            is_substring, following = self.query_helper(u_prime, x_prime, 0)
            if following == 'E': following = x[-1]
            if is_substring:
                if following == symbol:
                    sum_of_x += y * 1
                else:
                    sum_of_x += y * (-1/(s-1))

        return sum_of_x/len(self.labels)


if __name__ == '__main__':
    learner = ShuffleLearner()
    learner.learn_u()