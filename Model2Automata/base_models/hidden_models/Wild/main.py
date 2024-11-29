# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
# import numpy as np
# from base_models.hidden_models.Wild.Model import WildModel
# from base_models.hidden_models.Wild.train_data.data import get_train_test_data
# import matplotlib.pyplot as plt
#
# x_train, y_train, x_test, y_test = get_train_test_data()
#
# # X_train = np.concatenate(x_train)
# # X_test = np.concatenate(x_test)
# # Y_train = np.concatenate(y_train)
# # Y_test = np.concatenate(y_test)
#
#
# m = WildModel()
# m.build_model(m=4, h=4)
# m.fit(x_train, y_train, x_test, y_test)
# m.save_model('Wild_{}_{}'.format(4, 4))
#
# m1 = WildModel()
# m1.load_model('Wild_{}_{}'.format(4, 4))
#
# pred = m.predict_x_to_y(x_test[0])
# plt.scatter(y_test[0], pred, s=0.8)
# plt.show()
#
# hidden = m.predict_x_to_hidden(x_test[0])
# y = m.predict_hidden_to_y(x_test[0], hidden)
#
# plt.scatter(y, pred, s=0.8)
# plt.show()
#
