import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv

big_letters_const = 65


def ltr(i):
    return chr(big_letters_const + i)


class Discretizer:
    def __init__(self, continuous_data, N_clusters=15, mode = 'ltr'):
        self.k_means_hyp = KMeans(n_clusters=N_clusters, n_init=12, random_state=10)
        self.k_means_hyp.fit(continuous_data)
        self.clusters_to_centers = {}
        self.length = continuous_data.shape[1]

        if mode == 'ltr':
            self.index_func = ltr
        else:
            self.index_func = lambda x: str(x)

        for i in range(N_clusters):
            self.clusters_to_centers[self.index_func(i)] = self.k_means_hyp.cluster_centers_[i]

        self.clustered_data = self.k_means_hyp.predict(continuous_data.values)

        a = [self.index_func(i) for i in self.clustered_data]
        self.combined = pd.DataFrame(continuous_data, copy=True)
        self.combined['Cluster'] = a

        self.properties = {}
        for c in [self.index_func(i) for i in range(N_clusters)]:
            self.properties[c] = self.get_cluster_properties(c)

    def predict(self, X, mode):
        labeled = self.k_means_hyp.predict(X)
        res = []
        for x in labeled:
            if mode == 'disc':
                res.append(self.clusters_to_centers[self.index_func(x)])
            else:
                res.append(self.index_func(x))
        return res

    def get_cluster_properties(self, c):
        cluster_full_data = self.combined[self.combined['Cluster'] == c]
        d = {}
        for column in cluster_full_data.columns:
            if column != 'Cluster':
                vals = cluster_full_data[column].values
                d[column] = [np.min(vals), np.max(vals)]

        return d

    # def visualize(self, combined=None):
    #     if combined is None:
    #         combined = self.combined
    #
    #     if combined.shape[1] == 4:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(xs=combined.loc[:,combined.columns[0]],
    #                     ys=combined.loc[:,combined.columns[1]],
    #                     zs=combined.loc[:,combined.columns[2]],
    #                     c=combined['CLUSTER'], s=50)
    #         plt.show()
    #
    #     if combined.shape[1] == 2:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.scatter(x=combined.loc[:,combined.columns[0]],
    #                     y=[0]*combined.shape[0],
    #                     c=combined['CLUSTER'], s=50)
    #         plt.show()
