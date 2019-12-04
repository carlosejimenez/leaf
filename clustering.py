import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Cluster:
    def __init__(self, client_ids, gradients):
        from sklearn.cluster import OPTICS
        clusterer = OPTICS()
        self.cluster_tuple = cluster_gradients(client_ids, gradients, clusterer)

    def get_cluster(self, client_id):
        return self.cluster_tuple[3][client_id]


def cluster_gradients(gradient_ids, gradients, clusterer, projector=PCA(n_components=2), title="PCA Projection of Gradients. Colored by cluster.", cmap='hsv'):
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

    for client_num in range(len(c_labels)):
        client_to_cluster_dictionary[gradient_ids[client_num]] = c_labels[client_num]

    return C, c_labels, clusterer, client_to_cluster_dictionary


# def __test_clustering():
#     from sklearn.datasets import load_digits
#     from sklearn.cluster import OPTICS
#     digits = load_digits()
#     data = digits.data
#     ids = np.arange(data.shape[0])
#     clusterer = OPTICS()
#     cluster_gradients(ids, data, clusterer)
#
#
# if __name__ == '__main__':
#     __test_clustering()
