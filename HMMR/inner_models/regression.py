import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def create_vander_matrix(input_mat, p):
    poly = PolynomialFeatures(degree=p)
    vander_matrix = poly.fit_transform(input_mat)
    return vander_matrix


class Regression:
    def __init__(self, init_emissions_data):
        self.number_of_features = 0
        self.p = init_emissions_data['p'].values[0]
        self.sigma = init_emissions_data['sigma'].values[0]

        W = pd.DataFrame(init_emissions_data[init_emissions_data != '-'].drop(columns=['sigma', 'p']), dtype=np.float)
        W = W.values.flatten()
        self.init_coef(W)

    def init_coef(self, W):
        sample = np.random.normal(0, 1, size=W.size)
        self.regression = LinearRegression(fit_intercept=False).fit([sample], [8])
        self.regression.__dict__['coef_'] = W

    def predict_clean(self, data):
        if data.size == 0:
            return []
        vander_data = create_vander_matrix(data, self.p)

        clean_predict = self.regression.predict(vander_data)
        return clean_predict

    def predict_noisy(self, data):
        clean = self.predict_clean(data)
        noisy = clean + np.random.normal(0, self.sigma, size=data.shape[0])
        return noisy

    def get_log_prob_of_seq(self, in_seq, y_seq):
        vander_data = create_vander_matrix(in_seq, self.p)
        clean_predict = self.regression.predict(vander_data)
        probabilities = norm.logpdf(y_seq, clean_predict, self.sigma)
        return probabilities

    def update_regression(self, in_seq, y_seq, log_gamma):
        W = np.exp(log_gamma)
        # W[log_gamma > 0.6*np.min(log_gamma)] = 0.000000000000000000000001

        X = create_vander_matrix(in_seq, self.p)
        Y = y_seq.values

        self.regression.fit(X, Y, sample_weight=W)

        self.sigma = np.linalg.norm(np.sqrt(W) * (self.regression.predict(X) - Y))**2 / np.sum(W)

    def get_regression_for_file(self):
        return {'coef' :list(self.regression.__dict__['coef_']),
                'sigma' : self.sigma,
                'p': self.p
                }