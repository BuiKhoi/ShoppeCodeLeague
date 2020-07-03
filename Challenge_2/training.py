import os
import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import *

from DataGenerators import *

TRAIN_FOLDER = './shopee-product-detection-dataset/train/train/'
VAL_FOLDER = './shopee-product-detection-dataset/train/val/'
MODEL_CHECKPOINT = './Checkpoints/mobilenet_detect.h5'
MODEL_CHECKPOINT_LOSS = './Checkpoints/mobilenet_detect_loss.h5'

LOAD_WEIGHTS = True
WEIGHT_FILE = './Checkpoints/mobilenet_detect.h5'

def construct_model(num_class):
    base_model = MobileNetV2(include_top=False, weights=None, input_shape=(224,224,83))
    x=base_model.output
    # Add some new Fully connected layers to 
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x = Dropout(0.25)(x)
    x=Dense(512,activation='relu')(x) 
    x = Dropout(0.25)(x)
    preds=Dense(num_class, activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)
    return model

print('Constructing model')
train_model = construct_model(42)
train_model.summary()

if LOAD_WEIGHTS:
    train_model.load_weights(WEIGHT_FILE)
    print('Loaded pretrained weights')

# train_gen = NumpyDataGenerator(TRAIN_FOLDER, shuffle = True, batch_size=52)
# val_gen = NumpyDataGenerator(VAL_FOLDER, shuffle = False, batch_size=16)

train_gen = AugDataGenerator(TRAIN_FOLDER, shuffle = True, batch_size=52)
val_gen = AugDataGenerator(VAL_FOLDER, shuffle = False, batch_size=16)

reducelr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    MODEL_CHECKPOINT,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)

checkpoint_loss = ModelCheckpoint(
    MODEL_CHECKPOINT_LOSS,
    monitor="loss",
    verbose=1,
    save_best_only=True,
)

train_model.compile(optimizer=Adam(1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
# train_model.compile(optimizer=SGD(1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])

history = train_model.fit_generator(
    train_gen, 
    train_gen.__len__(), 
    200,
    validation_data=val_gen, 
    validation_steps=200,
    callbacks = [reducelr, checkpoint, checkpoint_loss]
)

plt.plot(history.history['val_accuracy'])
plt.show()