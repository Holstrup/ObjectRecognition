from current_model import new_function
from KNN_database_comparison import knn
import database_actions
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def load_model_from_file(filepath):
    """
    Loads CNN to global variable 'model'

    :param filepath: filepath to model file:
    """
    global model
    model = load_model(filepath)
    model.summary()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])


def show_image(image):
    """
    Simple function that displays the image. Can be used for verification purposes

    :param image: image array
    """
    plt.imshow(image)
    plt.show()



def build_db(x_data, y_data):
    """
    Function that builds a register of encodings and labels in the database
    NOTE: It reinitializes the database as well, so prior data will be deleted

    :param x_data: x_test loaded from an mnist subset
    :param y_data: y_test loaded from an mnnist subset
    """
    predictions = model.predict(x_data, batch_size=None, verbose=1, steps=None)
    database_actions.reinitialize_table()

    for i in range(len(predictions)):
        encoding = predictions[i]
        label = (np.where(y_data[i] == 1)[0]).item()
        database_actions.add_encoding(encoding, str(label))



def try_classification(data_no_label):
    """
    :param data_no_label: Image data without labels
    :return: Encodings for the image data
    """
    encodings = model.predict(data_no_label, batch_size=None, verbose=1, steps=None)
    return encodings


load_model_from_file("models/model09-02-2019-14:29")
(x_test, y_test), input_shape = new_function(28, 28, 10)
build_db(x_test[0:20], y_test[0:20])
data = try_classification(x_test[100:200])

tt = data[0].reshape(1, -1)
print(y_test[30:40])
knn(tt)
