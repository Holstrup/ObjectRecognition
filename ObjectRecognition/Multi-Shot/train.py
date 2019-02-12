import datetime
from model import conv_net, load_mnist_preprocessed_subset
from keras.models import Model


def run_train(n_epochs, model_name="null"):
    """
    Train our CNN

    :param n_epochs: Number of epochs
    :param model_name: file name of model. If nothing it will be the datetime
    """
    if model_name == "null":
        date = datetime.datetime.today().strftime('%d-%m-%Y-%H:%M')
        name = date
    else:
        name = model_name


    batch_size = 128
    num_classes = 5
    epochs = n_epochs
    img_rows, img_cols = 28, 28

    (x_train, y_train), input_shape = load_mnist_preprocessed_subset(img_rows, img_cols, num_classes)
    model = conv_net(num_classes, input_shape)
    model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1)
    CNN = Model(input = model.layers[0].input, output = model.layers[-3].output)
    CNN.summary()
    CNN.save("models/model" + name)

