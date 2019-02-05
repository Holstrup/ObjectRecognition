import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Average
from keras.optimizers import sgd
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical


def neural_network(n_classes):
    """
    :param n_classes: Number of classes in the dataset
    :return: A model that can classify these classes
    """
    num_classes = n_classes

    # Input - Layer
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))

    # Hidden - Layer

    # Output - Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Wrap up
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer=sgd(),
                  metrics=[categorical_accuracy])
    return model


def data_shaping(training_x, test_x, training_y, test_y, n_classes):
    """
    :param training_x: Training data
    :param test_x:  Test data
    :param training_y: Training classification
    :param test_y: Test classification
    :param n_classes: Number of classes
    :return: Same (MNist) data, but in the correct format
    """
    training_x = training_x.reshape(60000, 784)
    test_x = test_x.reshape(10000, 784)
    training_x = training_x.astype('float32')
    test_x = test_x.astype('float32')
    training_x /= 255
    test_x /= 255
    print(training_x.shape[0], 'train samples')
    print(test_x.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    training_y = to_categorical(training_y, num_classes=n_classes)
    test_y = to_categorical(test_y, num_classes=n_classes)
    return training_x, test_x, training_y, test_y


def data_shaping_subset(training_x, test_x, training_y, test_y, n_classes):
    """
    :param training_x: Training data
    :param test_x:  Test data
    :param training_y: Training classification
    :param test_y: Test classification
    :param n_classes: Number of classes
    :return: Subset of the (MNist) data, in the correct format
    """
    train_filter = np.where(training_y < n_classes - 1 )
    test_filter = np.where(test_y < n_classes - 1)

    training_x, training_y = training_x[train_filter], training_y[train_filter]
    test_x, test_y = test_x[test_filter], test_y[test_filter]

    print(training_x.shape)

    training_x = training_x.reshape(41935, 784)
    test_x = test_x.reshape(6989, 784)
    training_x = training_x.astype('float32')
    test_x = test_x.astype('float32')
    training_x /= 255
    test_x /= 255


    print(training_x.shape[0], 'train samples')
    print(test_x.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    training_y = to_categorical(training_y, num_classes=n_classes)
    test_y = to_categorical(test_y, num_classes=n_classes)
    return training_x, test_x, training_y, test_y