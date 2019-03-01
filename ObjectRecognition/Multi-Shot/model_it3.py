import numpy as np
import gzip
import tensorflow as tf
from keras.datasets import mnist

IMAGE_SIZE = 28
TEST_NUM_IMAGES = 60000

def load(file_name, num_examples):
    with gzip.open(file_name) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_examples)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_examples, IMAGE_SIZE, IMAGE_SIZE)
        return data

#Making 28X28 train matrix image into 30X30 using padding
def imagePadding(Input_Matrix):
        #Appending two columns of 0's at the end of 28X28 column matrix now matrix is of 28X30
        Desired_Matrix = np.zeros((28, 30))
        Desired_Matrix[:,:-2] = Input_Matrix
        No_Column_Input_Matrix=np.size(Desired_Matrix,1)
        if(No_Column_Input_Matrix==30):
            Extra_row=np.zeros(30)
        #appending row's with zeros at the end of Matrix
        for i in range(28):
            Desired_Matrix = np.vstack([Desired_Matrix, Extra_row])
        index=[]

        #Shifting rows according to index
        for i in range(29):
            index.append(i)
        temp=[29]
        final_index=temp+index
        Column_Exchange=Desired_Matrix[:,final_index]
        padding=Column_Exchange[final_index,:]
        return padding

# Implementation 1
def reshape():
    (x_train, label), (_, _) = mnist.load_data()

    batch_tensor = tf.reshape(x_train, [60000, 28, 28, 1])
    resized_images = tf.image.resize_images(batch_tensor, [30, 30])

    print np.shape(resized_images)

reshape()

# Implementation 2
train_images = load("{}/train-images-idx3-ubyte.gz".format("MNIST"), TEST_NUM_IMAGES)
new_mnist = np.zeros((60000, 30, 30))
for i in range(60000):
    new_image = imagePadding(train_images[i,:,:])
    new_mnist[i] = new_image

print np.shape(new_mnist)

