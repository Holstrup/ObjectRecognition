from sklearn.neighbors import NearestNeighbors
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


    dif_matrix = np.subtract(np.transpose(matrix_encodings), vector_encoding)
    norm_vector = np.linalg.norm(dif_matrix, axis=1)
    predicted_label = int(labels[np.argmin(norm_vector)])
    return predicted_label