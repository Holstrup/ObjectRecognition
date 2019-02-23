# Object Recognition
One-Shot object recognition thesis project by Alexander Holstrup and Cristian Botezatu

## Introduction
Object recognition is the subfield of computer vision that aims to solve the problem of detecting one or more objects in an image, providing not only the class of the objects but the position of them in that coordinate frame. 
The objective of this project would be to create a model that can perform one shot object recognition. I.e. Given an object recognition dataset with n classes C we aim to create a model that given a single image of class Cn+1, which the model has never seen before, it would be able to find this class on new samples. However, as a start, we will allow our model to get trained as much as needed, so that the accuracy could be increased. Afterwards, by maintaining the high accuracy, the number of trainings will be reduced until, eventually, reaching the one-shot object recognition.

## Performance metrics
All the metrics described below are implemented in sklearn.metrics.

### Adjusted Rand Index (ARI)
- Assume that true labels of objects are known.
- ARI does not depend on the labels' values but on the data cluster split.
- The evaluation is based on a share of observations for which the splits (initial and clustering result) are consistent.
- The Rand Index (RI) evaluates the similarity of the two splits of the same sample.
- This metric is symmetric and does not depend in the label permutation and, therefore, this index is a measure of distances between different sample splits (the range is [-1,1]).

### Adjusted Mutual Information (AMI)
- Likewise ARI, AMI is also symmetric and does not depend on the labels' values and permutation.
- AMI is defined by the entropy function and interprets a sample split as a discrete distribution (likelihood of assigning to a cluster is equal to the percent of objects in it).
- Intuitively, the mutual information measures the share of information common for both clustering splits (the range is [0,1]).

### Homogeneity, completeness, V-measure
- Formally, these metrics are also defined based on the entropy function and the conditional entropy function, interpreting the sample splits as discrete distributions.
- These metrics are not symmetric (the range is [0,1]).
- These metrics' values are not scaled as the  ARI  or  AMI  metrics are and, thus, depend on the number of clusters.
- V-measure is a combination of homogeneity, and completeness and is their harmonic mean. 
- V-measure is symmetric and measures how consistent two clustering results are.

### Silhouette
- Silhouette does not imply the knowledge about the true labels of the objects.
- It lets us estimate the quality of the clustering using only the initial, unlabeled sample and the clustering result.
- The silhouette distance shows to which extent the distance between the objects of the same class differ from the mean distance between the objects from different clusters.
- The higher the silhouette value is, the better the results from clustering (the range is [-1,1]).
- With the help of silhouette, we can identify the optimal number of clusters  k  (if we don't know it already from the data) by taking the number of clusters that maximizes the silhouette coefficient.

## Optimizations

High-dimensional datasets can be very difficult to visualize. While data in two or three dimensions can be plotted to show the inherent structure of the data, equivalent high-dimensional plots are much less intuitive. To aid visualization of the structure of a dataset, the dimension must be reduced in some way.

### Manifold learning (generalize linear frameworks to be sensitive to non-linear structure in data):

#### Isomap
Isomap can be viewed as an extension of Multi-dimensional Scaling (MDS) or Kernel PCA. Isomap seeks a lower-dimensional embedding which maintains geodesic distances between all points.

#### Local linear embedding
Locally linear embedding (LLE) seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods. It can be thought of as a series of local Principal Component Analyses which are globally compared to find the best non-linear embedding.

#### Modified local linear embedding
One well-known issue with LLE is the regularization problem. When the number of neighbors is greater than the number of input dimensions, the matrix defining each local neighborhood is rank-deficient. One method to address the regularization problem is to use multiple weight vectors in each neighborhood. This is the essence of modified locally linear embedding (MLLE).

#### Hessian eigenmapping
Hessian Eigenmapping (known also as Hessian-based LLE) is another method of solving the regularization problem of LLE. It revolves around a hessian-based quadratic form at each neighborhood which is used to recover the locally linear structure. Though other implementations note its poor scaling with data size, sklearn implements some algorithmic improvements which make its cost comparable to that of other LLE variants for small output dimension.

#### Spectral embedding
Spectral Embedding is an approach to calculating a non-linear embedding. Scikit-learn implements Laplacian Eigenmaps, which finds a low dimensional representation of the data using a spectral decomposition of the graph Laplacian. The graph generated can be considered as a discrete approximation of the low dimensional manifold in the high dimensional space. Minimization of a cost function based on the graph ensures that points close to each other on the manifold are mapped close to each other in the low dimensional space, preserving local distances.

#### Local Tangent Space Alignment
Though not technically a variant of LLE, Local tangent space alignment (LTSA) is algorithmically similar enough to LLE that it can be put in this category. Rather than focusing on preserving neighborhood distances as in LLE, LTSA seeks to characterize the local geometry at each neighborhood via its tangent space and performs a global optimization to align these local tangent spaces to learn the embedding.

#### Multi-dimensional scaling (MDS)
Multi-dimensional scaling (MDS) seeks a low-dimensional representation of the data in which the distances respect well the distances in the original high-dimensional space. In general, is a technique used for analyzing similarity or dissimilarity data. MDS attempts to model similarity or dissimilarity data as distances in a geometric space.

#### t-distributed Stochastic Neighbor Embedding (t-SNE)
While Isomap, LLE and variants are best suited to unfold a single continuous low dimensional manifold, t-SNE will focus on the local structure of the data and will tend to extract clustered local groups of samples as highlighted on the S-curve example. This ability to group samples based on the local structure might be beneficial to visually disentangle a dataset that comprises several manifolds at once as is the case in the digits dataset.

#### Random forest embedding
A forest embedding is a way to represent a feature space using a random forest. In our case, we are dealing with an unsupervised transformation of a dataset to a high-dimensional sparse representation. A data point is coded according to which leaf of each tree it is sorted into. Using a one-hot encoding of the leaves, this leads to a binary coding with as many ones as there are trees in the forest.

### PCA (linear framework not sensitive to non-linear structure in data):
Principal Component Analysis (PCA) is a dimension-reduction tool that can be used to reduce a large set of variables to a small set that still contains most of the information in the large set.
Objectives of principal component analysis:
1. PCA reduces attribute space from a larger number of variables to a smaller number of factors and as such is a "non-dependent" procedure (that is, it does not assume a dependent variable is specified).
2. PCA is a dimensionality reduction or data compression method. The goal is dimension reduction and there is no guarantee that the dimensions are interpretable (a fact often not appreciated by (amateur) statisticians).
3. To select a subset of variables from a larger set, based on which original variables have the highest correlations with the principal component.


## Setup

Developed in Python 2

### Install Libraries Needed
```
pip install -r  requirements.txt
```


## References


## Acknowledgements
Thank you to [Uizard Technologies](https://uizard.io/) for guidance through the project and to professor [Ole Winther](https://www.dtu.dk/english/service/phonebook/person?id=10167&cpid=109334&tab=3&qt=dtuprojectquery#tabs) from [Technical University of Denmark](https://www.dtu.dk).
