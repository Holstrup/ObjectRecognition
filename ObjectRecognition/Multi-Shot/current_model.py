import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Activation
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras import backend as K


def conv_net(num_classes, input_shape):
    """
    :param input_shape: Input shape of image
    :param num_classes: number of classes to classify in
    :return: A model that can classify these classes
    """
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))


    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    model.summary()
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    return model


def load_mnist_preprocessed(img_rows, img_cols, num_classes):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':  # Theano backend
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # normalise:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape


def load_mnist_preprocessed_subset(img_rows, img_cols, num_classes):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (_, _) = mnist.load_data()

    train_filter = np.where((y_train < 8))
    x_train, y_train = x_train[train_filter], y_train[train_filter]


    if K.image_data_format() == 'channels_first':  # Theano backend
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # normalise:
    x_train = x_train.astype('float32')
    x_train /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)

    return (x_train, y_train), input_shape

def new_function(img_rows, img_cols, num_classes):
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (_, _) = mnist.load_data()

    train_filter = np.where((y_train >= 8))
    x_train, y_train = x_train[train_filter], y_train[train_filter]


    if K.image_data_format() == 'channels_first':  # Theano backend
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # normalise:
    x_train = x_train.astype('float32')
    x_train /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)

    return (x_train, y_train), input_shape

