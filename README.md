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



## Setup

Developed in Python 2

### Install Libraries Needed
```
pip install -r  requirements.txt
```


## References


## Acknowledgements
Thank you to [Uizard Technologies](https://uizard.io/) for guidance through the project and to professor Winther from Technocal University of Denmark
