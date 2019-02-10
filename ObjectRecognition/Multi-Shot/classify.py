from current_model import mnist_test_set
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

database_actions.reinitialize_table()
load_model_from_file("models/model10-02-2019-12:31")
(x_test, y_test), input_shape = mnist_test_set(28, 28, 10)


#20 samples in the database
build_db(x_test[0:20], y_test[0:20])

#Testing on 1000 (different) samples
data = try_classification(x_test[100:1100])

correct = 0
wrong = 0

for i in range(len(data)):
    predicted_label = int(knn(data[i].reshape(1, -1)))
    real_label_one_hot = y_test[100 + i]
    real_label_int = (np.where(real_label_one_hot == 1)[0]).item()
    if real_label_int == predicted_label:
        print("Correct")
        correct += 1
    else:
        print "Wrong"
        wrong += 1

print correct

