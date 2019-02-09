from keras.models import load_model
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from Model import load_mnist_preprocessed
import datetime
import numpy as np
import matplotlib.pyplot as plt


def load_model_from_file(filepath):
    loaded_model = load_model(filepath)
    loaded_model.summary()
    return loaded_model

def show_image(image):
    plt.imshow(image)
    plt.show()

def get_image():
    (x_train, y_train), (_, _) = mnist.load_data()
    image_index = np.where(y_train == 9)[0][0]
    image = x_train[image_index]
    #show_image(image)
    return image



model = load_model_from_file("models/model08-02-2019-14:56")
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])


(x_train, y_train), (x_test, y_test), input_shape = load_mnist_preprocessed(28, 28, 10)


predictions = model.predict(x_test, batch_size=None, verbose=1, steps=None)
print(predictions[0].shape())