from database_actions import get_known_encodings
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
from keras.utils import np_utils
import numpy as np


def cov_network(num_classes):

    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 1), padding="same", activation="relu", input_shape=(1, 128, 1)))
    cnn.add(Conv2D(32, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(1, 1)))

    cnn.add(Conv2D(64, (3, 1), padding="same", activation="relu"))
    cnn.add(Conv2D(64, (3, 1), padding="same", activation="relu"))
    cnn.add(Conv2D(64, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(1, 1)))

    cnn.add(Conv2D(32, (3, 1), padding="same", activation="relu"))
    cnn.add(Conv2D(32, (3, 1), padding="same", activation="relu"))
    cnn.add(Conv2D(32, (3, 1), padding="same", activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(1, 1)))

    cnn.add(Flatten())
    cnn.add(Dense(1024, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(num_classes, activation="softmax"))

    cnn.compile(loss="categorical_crossentropy", optimizer="adam")
    cnn.summary()

    return cnn

def train_classify_network(EPOCHS):
    #Load data
    (encodings, labels) = get_known_encodings()
    x_train = encodings.astype("float32")
    X_shape = x_train.shape
    labels = np.array(labels).astype(int)
    num_classes = len(np.unique(labels))

    #Reformat X and Y
    x_train = x_train.reshape((X_shape[1], 1, X_shape[0], 1))
    labels = np_utils.to_categorical(labels)

    #Load Model
    model = cov_network(num_classes+1)

    #Train
    model.fit(x_train, labels, epochs=EPOCHS)
    model.save('models/classifier-model')



train_classify_network(1)
