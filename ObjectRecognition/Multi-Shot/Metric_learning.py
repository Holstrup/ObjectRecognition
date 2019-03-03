import sklearn.decomposition as deco
import numpy as np
import matplotlib.pyplot as plt
from database_actions import get_known_encodings
from sklearn import manifold
import metric_learn

# Load the data and define the necessary variables
X, labels = get_known_encodings()
X = np.transpose(X)

# Plot the dimensionality reduction frameworks.
def plot_2D(X, title=None):
    X = (X - np.mean(X, 0)) / np.std(X, 0)

    plt.figure()
    scatter_x = X[:, 0]
    scatter_y = X[:, 1]
    group = map(int, labels)
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
algorithms = ["lmnn", "itml", "nca", "sdml"]

algorithmDef = {
    "lmnn": metric_learn.LMNN(k=3, learn_rate=1e-6).fit_transform(X, labels),
    "itml": metric_learn.ITML_Supervised().fit_transform(X, labels),
    "nca": metric_learn.NCA(learning_rate=1e-6).fit_transform(X, labels),
    "sdml": metric_learn.SDML_Supervised().fit_transform(X, labels)
}

# Define the titles
title = {
    "lmnn": "LMNN",
    "itml": "ITML",
    "nca": "NCA",
    "sdml": "SDML"
}

# Plot the data points for which different algorithms are applied
for algo in algorithms:
    plot_2D(algorithmDef[algo], title[algo])
    plt.show()
