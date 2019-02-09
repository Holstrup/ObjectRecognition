from sklearn.neighbors import NearestNeighbors
from database_actions import *

def knn(encodings):

    samples = get_known_encodings()[0].transpose()
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(samples)
    i = neigh.kneighbors(encodings, return_distance=False)
    label = get_known_encodings()[1][i[0][0]]
    print(label)

knn([np.zeros(128)])