from keras import Sequential
from database_actions import get_known_encodings
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Dropout, Activation
from keras.models import Model

def cov_network(num_classes, input_shape):
    model = Sequential()

    model.add(Dense(128, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    return model

def run_conv_network(n_epochs):
    batch_size = 128
    num_classes = 5
    epochs = n_epochs
    input_shape = (128,20)

    (encodings, labels) = get_known_encodings()
    model = cov_network(num_classes, input_shape)
    model.fit(encodings, labels, batch_size=batch_size,epochs=epochs,verbose=1)
    CNN = Model(input = model.layers[0].input, output = model.layers[-5].output)
    CNN.summary()

run_conv_network(1)