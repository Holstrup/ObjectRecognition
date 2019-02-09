from sklearn.neighbors import NearestNeighbors
from database_actions import *


def knn(encodings):
    """
    Once being trained, the already existent data will be clustered corresponding to their nature.
    Thus, by comparing the encodings and finding the 1 nearest neighbour, we will assign the new sample to the proper cluster via its label.
    """
    samples = get_known_encodings()[0].transpose()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    i = neigh.kneighbors(encodings, return_distance=False)
    label = get_known_encodings()[1][i[0][0]]
    return label

