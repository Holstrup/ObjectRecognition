import math, os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.utils import plot_model

DATA_DIR = 'Dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 1


num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

gen = ImageDataGenerator()
val_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical',
                                  shuffle=True, batch_size=BATCH_SIZE)
val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical',
                                          shuffle=True, batch_size=BATCH_SIZE)

model = ResNet50(weights='imagenet')

classes = list(iter(batches.class_indices))
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
last = model.layers[-1].output
x = Dense(128, activation="softmax")(last)
x = Dense(len(classes), activation="softmax")(x)
finetuned_model = Model(model.input, x)
finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
finetuned_model.summary()

for c in batches.class_indices:
    classes[batches.class_indices[c]] = c
finetuned_model.classes = classes

early_stopping = EarlyStopping(patience=10)
checkpointer = ModelCheckpoint('models/resnet50_best.h5', verbose=1, save_best_only=True)

finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCHS,
                              callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                              validation_steps=num_valid_steps)

finetuned_model.save('models/resnet50_finetuned')

for layer in model.layers:
    layer.trainable = True

finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

finetuned_model.summary()

finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=EPOCHS,
                              callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                              validation_steps=num_valid_steps)

finetuned_model.save('models/resnet50_fully_trained')



""" 
plot_model(finetuned_model,to_file='demo.png',show_shapes=True)
"""