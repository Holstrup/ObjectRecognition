from keras.datasets import mnist
from sklearn import preprocessing
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

(train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
dim_x = train_xs.shape[1] * train_xs.shape[2]
dim_y = 10


train_xs = train_xs.reshape(train_xs.shape[0], dim_x).astype(np.float32)

scaler = preprocessing.MinMaxScaler().fit(train_xs)
train_xs = scaler.transform(train_xs)
print(train_xs.shape)
print(train_ys.shape)

ridx = np.random.randint(train_xs.shape[0], size=1000)
np_train_xs = train_xs[ridx, :]
np_train_ys = train_ys[ridx]
print(np_train_xs.shape)
print(np_train_ys.shape)

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


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
    plt.show()

tsne_train_xs = TSNE(random_state=42).fit_transform(np_train_xs)
draw_scatter(tsne_train_xs, dim_y, np_train_ys)