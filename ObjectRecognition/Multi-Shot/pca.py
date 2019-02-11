import pylab as plt
import seaborn as sns; sns.set()
import sklearn.decomposition as deco
from model import *
from database_actions import get_known_encodings
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pca_full_mnist():
    # (x_train, y_train), (x_test, y_test) = load_mnist_preprocessed(28, 28, 10)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255

    mu = x_train.mean(axis=0)
    U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
    Zpca = np.dot(x_train - mu, V.transpose())

    Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu
    err = np.sum((x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]

    plt.title('PCA')
    plt.scatter(Zpca[:5000,0], Zpca[:5000,1], c=y_train[:5000], s=8, cmap='tab10')
    plt.gca().get_xaxis().set_ticklabels([])
    plt.gca().get_yaxis().set_ticklabels([])
    plt.savefig('pca/pca_full.png')
    plt.show()



def pca():
    matrix_encodings, labels =  get_known_encodings()
    matrix_encodings = np.transpose(matrix_encodings)
    n_components = 3

    x = (matrix_encodings - np.mean(matrix_encodings, 0)) / np.std(matrix_encodings, 0)
    pca = deco.PCA(n_components)
    x_r = pca.fit(x).transform(x)
    return x_r, labels



def plot(data, labels):
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


    plt.subplots()
    ax = plt.axes(projection='3d')
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_z[ix], scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.savefig('pca/pca_current_encodings.png')
    plt.show()



pca_data, db_labels = pca()
plot(pca_data, db_labels)

