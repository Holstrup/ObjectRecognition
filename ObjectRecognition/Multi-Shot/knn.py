from sklearn.neighbors import NearestNeighbors
from database_actions import get_known_encodings
import numpy as np

def knn_function(encoding):
    """
    Once being trained, the already existent data will be clustered corresponding to their nature.
    Thus, by comparing the encodings and finding the 1 nearest neighbour, we will assign the new sample to the proper cluster via its label.

    :param encoding: Encoding of an image
    :return: Predicted label of the image
    """
    samples = get_known_encodings()[0].transpose()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    i = neigh.kneighbors(encoding, return_distance=False)
    label = get_known_encodings()[1][i[0][0]]
    return label



def new_knn_function(encodings):
    """
    Function that returns the nearest neighbor to an image encoding

    :param encodings: Vector of encoding for an image (128,)
    :return: Predicted label (int)
    """
    vector_encoding = encodings
    matrix_encodings, labels = get_known_encodings()


    dif_matrix = np.subtract(np.transpose(matrix_encodings), vector_encoding)
    norm_vector = np.linalg.norm(dif_matrix, axis=1)
    predicted_label = int(labels[np.argmin(norm_vector)])
    return predicted_label