import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from database_actions import get_known_encodings

def t_sne():
    """
        Does TSNE of dataset in the database
    """
    matrix_encodings, labels = get_known_encodings()
    matrix_encodings = np.transpose(matrix_encodings)
    n_components = 3

    x = (matrix_encodings - np.mean(matrix_encodings, 0)) / np.std(matrix_encodings, 0)
    tsne = TSNE(n_components, init='pca', random_state=0)
    x_r = tsne.fit_transform(x)
    return x_r, labels

def plot3D(data, labels):
    """
    Plots PCA with Labels

    :param data: PCA Data
    :param labels: Labels
    """
    scatter_x = data[:, 0]
    scatter_y = data[:, 1]
    scatter_z = data[:, 2]
    group = map(int, labels)
    cdict = {5: 'red', 6: 'blue', 7: 'green', 8: 'black', 9: 'orange'}


    plt.subplots()
    ax = plt.axes(projection='3d')
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.subplots()
    ax = plt.axes(projection='3d')
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_z[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.savefig("figures/tsne_current_encodings_3D")
    plt.show()

pca_data, db_labels = t_sne()
plot3D(pca_data, db_labels)