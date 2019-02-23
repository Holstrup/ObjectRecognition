from time import time
import sklearn.decomposition as deco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from database_actions import get_known_encodings
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

digits = get_known_encodings()
X, y = get_known_encodings()
X = np.transpose(X)
n_samples, n_features = X.shape
n_neighbors = 21

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
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

# #----------------------------------------------------------------------
# # Projection on to the first 12 priciple components
print("Computing Linear Discriminant Analysis projection")
t0 = time()
x = (X - np.mean(X, 0)) / np.std(X, 0)
pca = deco.PCA(n_components = 12)
x_pca = pca.fit(x).transform(x)
plot_embedding(x_pca,
               "Linear Discriminant projection of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/pca_discriminant")

# #----------------------------------------------------------------------
# # Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
print("Done.")
plot_embedding(X_iso,
               "Isomap projection of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/isomap")

# #----------------------------------------------------------------------
# # Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='standard')
t0 = time()
X_lle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle,
               "Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/local_linear_embedding")

# #----------------------------------------------------------------------
# # Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='modified')
t0 = time()
X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_mlle,
               "Modified Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/modified_local_linear_embedding")

# #----------------------------------------------------------------------
# # HLLE embedding of the digits dataset
print("Computing Hessian LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='hessian')
t0 = time()
X_hlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_hlle,
               "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/hlle")


# #----------------------------------------------------------------------
# # LTSA embedding of the digits dataset
print("Computing LTSA embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                      method='ltsa')
t0 = time()
X_ltsa = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_ltsa,
               "Local Tangent Space Alignment of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/ltsa")

# #----------------------------------------------------------------------
# # MDS  embedding of the digits dataset
print("Computing MDS embedding")
clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
t0 = time()
X_mds = clf.fit_transform(X)
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds,
               "MDS embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/mds")

# #----------------------------------------------------------------------
# # Random Trees embedding of the digits dataset
print("Computing Totally Random Trees embedding")
hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
t0 = time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.TruncatedSVD(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding(X_reduced,
               "Random forest embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/random_trees")

# #----------------------------------------------------------------------
# # Spectral embedding of the digits dataset
print("Computing Spectral embedding")
embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
t0 = time()
X_se = embedder.fit_transform(X)

plot_embedding(X_se,
               "Spectral embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.savefig("figures/Reduction_OurDataSet/spectral")

#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)


plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

plt.savefig("figures/Reduction_OurDataSet/tsne")
plt.show()