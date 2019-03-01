from keras import Model
from keras_preprocessing import image
from keras.models import load_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import os
from knn import new_knn_function
from model import mnist_test_set
import database_actions
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


def load_imagenet_model(filepath):
    """
    Loads fine-tuned imagenet to global variable 'imagenet_model'

    :param filepath: filepath to model file:
    """
    global imagenet_model
    model = load_model(filepath)
    imagenet_model = Model(input=model.layers[0].input, output=model.layers[-3].output)
    imagenet_model.summary()


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



def run_classify(model_name_time):
    """
    Runs a test of the network

    :param model_name_time: model name
    """
    database_actions.reinitialize_table()
    load_model_from_file("models/" + model_name_time)
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    (x_test, y_test), input_shape = mnist_test_set(28, 28, 10)


    #20 samples in the database
    build_db(x_test[0:100], y_test[0:100])

    #Testing on 1000 (different) samples
    data = try_classification(x_test[100:2100])

    correct = 0

    for i in range(len(data)):
        predicted_label = new_knn_function(data[i])
        real_label_int = (np.where(y_test[100 + i] == 1)[0]).item()
        if real_label_int == predicted_label:
            correct += 1
    print "Correct: " + str(correct)

def build_imagenet_db(model_name):
    """
    Builds the database from the model specified.

    :param model_name: File name of model file
    """
    database_actions.reinitialize_table()
    TEST_DIR = "Dataset/test/greek/"

    load_imagenet_model("models/" + model_name)
    imagenet_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    file_list = os.listdir(TEST_DIR)[1:]
    for imgFile in file_list:
        if imgFile[0] == ".":
            pass
        else:
            img = image.load_img(TEST_DIR + imgFile, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            encoding = imagenet_model.predict(img, batch_size=1)
            label = imgFile[0]
            database_actions.add_encoding(encoding, label)



def run_classify_imagenet(model_name):
    """
        Run Image-net classification
    """
    build_imagenet_db(model_name)
    VAL_DIR = "Dataset/test/validate/"
    file_list = os.listdir(VAL_DIR)
    for imgFile in file_list:
        img = image.load_img(VAL_DIR + imgFile, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        encoding = imagenet_model.predict(img, batch_size=1)
        print "Predicted label {} for image file: {}".format(new_knn_function(encoding), imgFile)


run_classify("model11-02-2019-11:59")
