import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from base_models.no_hidden_models.rhlp.Model import RHLPModel
from base_models.no_hidden_models.rhlp.train_data.data import get_train_test_data
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = get_train_test_data()

X_train = np.concatenate(x_train)
X_test = np.concatenate(x_test)
Y_train = np.concatenate(y_train)
Y_test = np.concatenate(y_test)


m = RHLPModel()
m.build_model(m=6, h=5, beta=1)
m.fit(X_train, Y_train, X_test, Y_test)
m.save_model('rhlp_{}_{}'.format(6, 5))

m1 = RHLPModel()
m1.load_model('rhlp_{}_{}'.format(6, 5))

# np.testing.assert_allclose(m.predict(X_test), m1.predict(X_test))

pred = m.predict_x_to_y(X_test)
plt.scatter(Y_test, pred, s=0.8)
plt.show()

pred1 = m1.predict_x_to_y(X_test)
# plt.scatter(Y_test, pred1, s=0.8)
# plt.show()

plt.scatter(pred, pred1, s=0.3)
plt.show()

