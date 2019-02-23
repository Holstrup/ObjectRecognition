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



# Projection on to the first 12 priciple components
print("Running PCA of 12 PC")
pca = deco.PCA(n_components = 12)
x_pca = pca.fit(X).transform(X)
plot_2D(x_pca, "Principal Components projection of the digits")
plt.savefig("figures/Reduction_OurDataSet/pca_discriminant")



# Isomap projection of the digits dataset
print("Running Isomap embedding")
iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
plot_2D(iso, "Isomap projection of the digits")
plt.savefig("figures/Reduction_OurDataSet/isomap")



# Locally linear embedding of the digits dataset
print("Running LLE embedding")
lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
X_lle = lle.fit_transform(X)
plot_2D(X_lle, "Locally Linear Embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/local_linear_embedding")



# Modified Locally linear embedding of the digits dataset
print("Running modified LLE embedding")
mlle = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
X_mlle = mlle.fit_transform(X)
plot_2D(X_mlle, "Modified Locally Linear Embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/modified_local_linear_embedding")



# HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
hes = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
X_hlle = hes.fit_transform(X)
plot_2D(X_hlle, "Hessian Locally Linear Embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/hlle")



# LTSA embedding of the digits dataset
print("Running LTSA embedding")
ltsa = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
X_ltsa = ltsa.fit_transform(X)
plot_2D(X_ltsa, "Local Tangent Space Alignment of the digits")
plt.savefig("figures/Reduction_OurDataSet/ltsa")



# MDS embedding of the digits dataset
print("Running MDS embedding")
mds = manifold.MDS(n_components=2, n_init=1, max_iter=100)
X_mds = mds.fit_transform(X)
plot_2D(X_mds, "MDS embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/mds")



# Spectral embedding of the digits dataset
print("Running Spectral embedding")
se = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
X_se = se.fit_transform(X)
plot_2D(X_se, "Spectral embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/spectral")



# t-SNE embedding of the digits dataset
print("Running t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)
plot_2D(X_tsne, "t-SNE embedding of the digits")
plt.savefig("figures/Reduction_OurDataSet/tsne")

plt.show()