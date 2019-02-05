import numpy as np
from keras.datasets import mnist
from Model import neural_network, data_shaping, data_shaping_subset
import datetime

# Get date
date = datetime.datetime.today().strftime('%d-%m-%Y')

# Parameters
batch_size = 128
n_classes = 8
epochs = 2

# Import mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshape data
"""x_train, x_test, y_train, y_test = dataShaping(x_train, x_test, y_train, y_test, n_classes)"""
x_train, x_test, y_train, y_test = data_shaping_subset(x_train, x_test, y_train, y_test, n_classes)
print(np.shape(x_train))


# Build neural network
model = neural_network(n_classes)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


model.save("models/model" + str(date))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
