import pylab as plt
import seaborn as sns; sns.set()
import sklearn.decomposition as deco
from model import *
from database_actions import get_known_encodings
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pca():
    """
    Does PCA of dataset in the database

    :return: PCA and labels
    """
    matrix_encodings, labels =  get_known_encodings()
    matrix_encodings = np.transpose(matrix_encodings)
    n_components = 20

    x = (matrix_encodings - np.mean(matrix_encodings, 0)) / np.std(matrix_encodings, 0)
    pca = deco.PCA(n_components)
    x_r = pca.fit(x).transform(x)

    plt.figure(figsize=(10, 7))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
    plt.xlabel('Number of components')
    plt.ylabel('Total explained variance')
    plt.xlim(0, 20)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.axvline(6.5, c='b')
    plt.axhline(0.9, c='r')
    plt.savefig("figures/pca_explained_variance")
    plt.show();

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
    plt.savefig("figures/pca_current_encodings_3D")
    plt.show()

pca_data, db_labels = pca()
plot3D(pca_data, db_labels)

