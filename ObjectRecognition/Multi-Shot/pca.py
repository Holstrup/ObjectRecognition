import pylab as plt
import numpy as np
import seaborn as sns; sns.set()
from keras.datasets import mnist
from current_model import *

# (x_train, y_train), (x_test, y_test) = load_mnist_preprocessed(28, 28, 10)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
Zpca = np.dot(x_train - mu, V.transpose())

Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu
err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]

plt.figure(figsize=(8,4))
plt.title('PCA')
plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])
plt.show()