import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from base_models.hidden_models.rhlp_hidden.Model import RHLPModel
from base_models.hidden_models.rhlp_hidden.train_data.data import get_train_test_data
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = get_train_test_data()

X_train = np.concatenate(x_train)
X_test = np.concatenate(x_test)
Y_train = np.concatenate(y_train)
Y_test = np.concatenate(y_test)


m_value = 6
h_values = list(range(3, 8))
beta = [1, 2, 5, 10]

f = 6
h = 5

model = RHLPModel()
model.build_model(m=f, h=h, beta=1)
model.fit(X_train, Y_train, X_test, Y_test)
model.save_model('rhlp_{}_{}'.format(f, h))

model = RHLPModel()
model.load_model('rhlp_{}_{}'.format(f, h))

pred = model.predict_x_to_y(X_test)
plt.scatter(Y_test, pred, s=0.8)
plt.show()

hidden = model.predict_x_to_hidden(X_test)
y = model.predict_hidden_to_y(X_test, hidden)

plt.scatter(y, pred, s=0.8)
plt.show()


import argparse
parser = argparse.ArgumentParser(description="Train RHLP Hidden")


if __name__ == '__main__':
    ...
