import datetime
from current_model import conv_net, load_mnist_preprocessed, load_mnist_preprocessed_subset
from keras.models import Model


date = datetime.datetime.today().strftime('%d-%m-%Y-%H:%M')
batch_size = 128
num_classes = 5
epochs = 4
img_rows, img_cols = 28, 28

#Dimensions: (60000, 28, 28, 1), (60000, 10), (10000, 28, 28, 1), (10000, 10)
#(x_train, y_train), (x_test, y_test), input_shape = load_mnist_preprocessed(img_rows, img_cols, num_classes)

(x_train, y_train), input_shape = load_mnist_preprocessed_subset(img_rows, img_cols, num_classes)



model = conv_net(num_classes, input_shape)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

CNN = Model(input = model.layers[0].input, output = model.layers[15].output)

CNN.summary()

CNN.save("models/model" + str(date))


