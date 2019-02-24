import sklearn.decomposition as deco
import numpy as np
import matplotlib.pyplot as plt
from database_actions import get_known_encodings
from sklearn import (manifold, decomposition, ensemble)

# Load the data and define the necessary variables
X, y = get_known_encodings()
X = np.transpose(X)
n_samples, n_features = X.shape
n_neighbors = 21

# Plot the dimensionality reduction frameworks.
def plot_2D(X, title=None):
    X = (X - np.mean(X, 0)) / np.std(X, 0)

    plt.figure()
    scatter_x = X[:, 0]
    scatter_y = X[:, 1]
    group = map(int, y)
    cdict = {5: 'red', 6: 'blue', 7: 'green', 8: 'black', 9: 'orange'}

    ax = plt.axes()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("X")
    plt.ylabel("Y")

    if title is not None:
        plt.title(title)

# Initiate the list of algorithms
algorithms = ["x_pca", "x_iso", "x_lle", "x_mlle", "x_hes", "x_ltsa", "x_mds", "x_se", "x_tsne"]

algorithmDef = {
    "x_pca": deco.PCA(n_components = 12).fit(X).transform(X),
    "x_iso": manifold.Isomap(n_neighbors, n_components=2).fit_transform(X),
    "x_lle": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard').fit_transform(X),
    "x_mlle": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified').fit_transform(X),
    "x_hes": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian').fit_transform(X),
    "x_ltsa": manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa').fit_transform(X),
    "x_mds": manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(X),
    "x_se": manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack").fit_transform(X),
    "x_tsne": manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
}

# Define the titles
title = {
    "x_pca": "Principal Components projection of the digits",
    "x_iso": "Isomap projection of the digits",
    "x_lle": "Locally Linear Embedding of the digits",
    "x_mlle": "Modified Locally Linear Embedding of the digits",
    "x_hes": "Hessian Locally Linear Embedding of the digits",
    "x_ltsa": "Local Tangent Space Alignment of the digits",
    "x_mds": "MDS embedding of the digits",
    "x_se": "Spectral embedding of the digits",
    "x_tsne": "t-SNE embedding of the digits"
}

# Plot the data points for which different algorithms are applied
for algo in algorithms:
    plot_2D(algorithmDef[algo], title[algo])
    plt.show()