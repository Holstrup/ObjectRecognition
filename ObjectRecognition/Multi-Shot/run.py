import datetime
from Model import conv_net, load_mnist_preprocessed
from keras.models import Model


date = datetime.datetime.today().strftime('%d-%m-%Y-%H:%M')
batch_size = 128
num_classes = 10
epochs = 1
img_rows, img_cols = 28, 28

#Dimensions: (60000, 28, 28, 1), (60000, 10), (10000, 28, 28, 1), (10000, 10)
(x_train, y_train), (x_test, y_test), input_shape = load_mnist_preprocessed(img_rows, img_cols, num_classes)



model = conv_net(num_classes, input_shape)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

CNN = Model(input = model.layers[0].input, output = model.layers[15].output)

CNN.summary()


CNN.save("models/model" + str(date))

score = CNN.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

