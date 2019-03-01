from database_actions import get_known_encodings
import numpy as np

def new_knn_function(encodings):
    """
    Function that returns the nearest neighbor to an image encoding

    :param encodings: Vector of encoding for an image (128,)
    :return: Predicted label (int)
    """
    vector_encoding = encodings
    matrix_encodings, labels = get_known_encodings()

    dist_vector = euclidean_distance(vector_encoding, matrix_encodings)
    predicted_label = int(labels[np.argmin(dist_vector)])

    return predicted_label



def euclidean_distance(vector, matrix):
    dif_matrix = np.subtract(np.transpose(matrix), vector)
    return np.linalg.norm(dif_matrix, axis=1)


def manhattan_distance(vector, matrix):
    dif_matrix = np.abs(np.subtract(np.transpose(matrix), vector))
    return dif_matrix.sum(axis = 1)

