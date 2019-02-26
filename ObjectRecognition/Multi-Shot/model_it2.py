import math, os
from keras.layers import Dense, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50

DATA_DIR = 'Dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1

print "-----INITIALIZING-----"
num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_train_steps = math.floor(num_train_samples / BATCH_SIZE)

gen = ImageDataGenerator()
val_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical',
                                  shuffle=True, batch_size=BATCH_SIZE)


model = ResNet50(weights='imagenet')
classes = list(iter(batches.class_indices))
model.layers.pop()
for layer in model.layers:
    layer.trainable = False

last = model.layers[-1].output
x = Dense(128, activation="relu")(last)
x = UpSampling1D(size=4)(x)
x = Dense(len(classes), activation="softmax")(x)
finetuned_model = Model(model.input, x)
finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
finetuned_model.summary()


for c in batches.class_indices:
    classes[batches.class_indices[c]] = c
finetuned_model.classes = classes

print "-----STARTING FIRST TRAINING-----"
finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCHS)

print "-----DONE WITH FIRST TRAINING-----"

for layer in finetuned_model.layers:
    finetuned_model.trainable = True

print "-----STARTING SECOND TRAINING-----"
finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCHS)


print "-----SAVING MODEL-----"
finetuned_model.save('models/resnet50-model')
print "-----DONE-----"

#plot_model(finetuned_model,to_file='demo.png',show_shapes=True)
