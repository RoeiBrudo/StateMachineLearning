from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def ltr(i):
    return 'S{}'.format(i)


class KMeansWrapper:
    def __init__(self, continuous_data, n_clusters, mode):
        self.dim_of_input = continuous_data.shape[1]
        self.clustering_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=12)
        self.clustering_model.fit(continuous_data)

        if mode == 'ltr':
            self.index_func = ltr
        else:
            self.index_func = lambda x: str(x+1)

        self.clusters_to_centers = {}
        for i in range(n_clusters):
            self.clusters_to_centers[self.index_func(i)] = self.clustering_model.cluster_centers_[i]

    def predict_from_vectors(self, X, to_vectors=True):
        clusters_indices = self.__k_means_prediction(X)
        clusters = [self.index_func(i) for i in clusters_indices]
        if to_vectors:
            X_representative = [self.clusters_to_centers[c] for c in clusters]
            return np.array(X_representative)
        else:
            return np.array(clusters)

    def get_clusters_representatives(self, X):
        X_representatives = [self.clusters_to_centers[x] for x in X]
        return np.array(X_representatives)

    def __k_means_prediction(self, X):
        if self.dim_of_input == 1:
            clusters_indices = self.clustering_model.predict(np.array(X).reshape((-1,1)).astype(np.float))
        else:
            clusters_indices = self.clustering_model.predict(X)

        return clusters_indices


def decide_num_of_clusters(max_range, continuous_data):
    clusters_range = np.arange(1, max_range)
    inertia = []
    for i in range(1, max_range):
        k_means_hyp = KMeans(n_clusters=i , n_init=12, random_state=10)
        k_means_hyp.fit(continuous_data)
        inertia.append(k_means_hyp.inertia_)

    plt.plot(clusters_range, inertia)
    plt.show()
