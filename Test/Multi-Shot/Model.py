from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical


def neural_network(n_classes):
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


def dataShaping(training_x, test_x, training_y, test_y, n_classes):
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
