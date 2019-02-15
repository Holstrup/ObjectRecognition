from keras.datasets import mnist
from sklearn import preprocessing
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from database_actions import get_known_encodings
from mpl_toolkits import mplot3d

def draw_scatter(x, n_class, colors):
    sns.palplot(sns.color_palette("hls", n_class))
    palette = np.array(sns.color_palette("hls", n_class))

    f = plt.figure(figsize=(14, 14))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')


def t_sne_full_mnist():
    """
        Generates plot of TSNE of entire MNist dataset
    """
    (train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
    dim_y = 10
    dim_x = train_xs.shape[1] * train_xs.shape[2]
    train_xs = train_xs.reshape(train_xs.shape[0], dim_x).astype(np.float32)
    scaler = preprocessing.MinMaxScaler().fit(train_xs)
    train_xs = scaler.transform(train_xs)

    ridx = np.random.randint(train_xs.shape[0], size=1000)
    np_train_xs = train_xs[ridx, :]
    np_train_ys = train_ys[ridx]

    return np_train_xs, np_train_ys, dim_y


def plot_tsne(np_train_xs, np_train_ys, dim_y):

    tsne_train_xs = TSNE(random_state=3).fit_transform(np_train_xs)
    draw_scatter(tsne_train_xs, dim_y, np_train_ys)
    plt.savefig("figures/tsne_full")
    plt.show()

np_train_xs, np_train_ys, dim_y = t_sne_full_mnist()
# plot_tsne(np_train_xs, np_train_ys, dim_y)

def t_sne():
    """
        Does TSNE of dataset in the database
    """
    matrix_encodings, labels = get_known_encodings()
    matrix_encodings = np.transpose(matrix_encodings)
    n_components = 3

    x = (matrix_encodings - np.mean(matrix_encodings, 0)) / np.std(matrix_encodings, 0)
    tsne = TSNE(n_components)
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

def plot2D(data, labels):
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

    ax = plt.axes()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.title("TSNE for the encodings saved in the database")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("figures/tsne_current_encodings_2D")
    plt.show()

pca_data, db_labels = t_sne()
plot3D(pca_data, db_labels)
plot2D(pca_data, db_labels)