from keras.models import load_model
from keras.datasets import mnist
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
    show_image(image)
    return image



date = datetime.datetime.today().strftime('%d-%m-%Y-%H:%M')
model = load_model_from_file("models/model" + str(date))
test_image = get_image().reshape(1, 784)
model.predict(test_image, batch_size=None, verbose=1, steps=None)