from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

TRAIN_FOLDER = './shopee-product-detection-dataset/train/train/'
VAL_FOLDER = './shopee-product-detection-dataset/train/val/'
MODEL_CHECKPOINT = './Checkpoints/inception.h5'

LOAD_WEIGHTS = True
WEIGHT_FILE = './Checkpoints/inception.h5'

def construct_model(num_class):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
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

train_gen = ImageDataGenerator(rotation_range=10)
val_gen = ImageDataGenerator(rotation_range=10)
train_gen = train_gen.flow_from_directory(TRAIN_FOLDER, (224, 224), batch_size = 16)
val_gen = val_gen.flow_from_directory(VAL_FOLDER, (224, 224), shuffle=False, batch_size=8)

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

# train_model.compile(optimizer=Adam(1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])
train_model.compile(optimizer=SGD(1e-4), loss = 'categorical_crossentropy', metrics=['accuracy'])

history = train_model.fit_generator(
    train_gen, 
    800, 
    200, 
    validation_data=val_gen, 
    validation_steps=50,
    callbacks = [reducelr, checkpoint]
)

plt.plot(history.history['val_accuracy'])
plt.show()