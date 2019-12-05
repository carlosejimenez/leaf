import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn.decomposition import PCA


class Clustering:
    def __init__(self, gradients, algorithm, num_clusters):
        self.algorithm = algorithm
        self.c_count = num_clusters #Number of clusters
        if algorithm == 'Affinity':
            self.clusterer = sklearn.cluster.AffinityPropagation()
        elif algorithm == 'Agglo':
            self.clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=num_clusters)
        elif algorithm == 'Birch':
            self.clusterer = sklearn.cluster.Birch(n_clusters=num_clusters)

        self.cluster_tuple = cluster_gradients(gradients, self.clusterer)

    def get_cluster(self, client_id):
        return self.cluster_tuple[3][client_id]

    def predict(self, gradient):
        #gradient = np.reshape(gradient, (1, len(gradient)))
        cluster_id = self.clusterer.predict(gradient)#[0]
        return cluster_id

    def number_clusters(self):
        if self.algorithm == 'Affinity':
            return self.clusterer.cluster_centers_indices_.shape[0]
        return self.c_count


def cluster_gradients(gradients, clusterer, projector=None, title="PCA Projection of Gradients. Colored by cluster.", cmap='hsv'):
    """
    Parameters:
        gradient_ids: np array. Shape: (num_clients,)
        graidents: np.array. Shape: (num_clients, gradient_dim)
        clusterer: sklearn.clustering object. MUST have fit_predict method
        projector: sklearn obj used for projection. MUST have fit_transform method. Results are not visualized with matplotlib if None
        title: Title of projection
    
    returns:
        tuple (C, c_labels, clusterer):
            C - number of classes. If based on non fixed number of clusters algo (DBSCAN or OPTICS) this returns the largest cluster id. Don't user this if you know number
                of classes beforehand. (You should already have the number of classes).
            c_labels - np.array representing class label per grad instance. Shape: (num_clients,)
            clusterer - returns the clustering object. Useful for extracting information about cluster centers etc.
    """

    client_to_cluster_dictionary = {}
    c_labels = clusterer.fit_predict(gradients)
    C = np.max(c_labels)
    if projector is not None:
        proj = projector.fit_transform(gradients)
        colors = c_labels / C
        plt.set_cmap(cmap)
        plt.scatter(proj[:, 0], proj[:, 1], c=colors)
        plt.show()

    return C, c_labels, clusterer, client_to_cluster_dictionary