import datetime
from Model import neural_network, load_mnist_preprocessed


date = datetime.datetime.today().strftime('%d-%m-%Y-%H:%M')
batch_size = 128
num_classes = 10
epochs = 2
img_rows, img_cols = 28, 28


(x_train, y_train), (x_test, y_test), input_shape = load_mnist_preprocessed(img_rows, img_cols, num_classes)

model = neural_network(num_classes, input_shape)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

model.save("models/model" + str(date))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])