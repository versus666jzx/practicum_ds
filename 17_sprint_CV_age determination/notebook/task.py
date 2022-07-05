import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np


def load_train(path):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True
    )

    labels = pd.read_csv(f'{path}/labels.csv')
    train_datagen = datagen.flow_from_dataframe(
        labels,
        path,
        batch_size=16,
        target_size=(150, 150),
        subset='training',
        class_mode='sparse',
        seed=12345)

    return train_datagen


def load_test(path):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True
    )

    labels = pd.read_csv(f'{path}/labels.csv')
    train_datagen = datagen.flow_from_dataframe(
        labels,
        path,
        batch_size=16,
        target_size=(150, 150),
        subset='training',
        class_mode='sparse',
        seed=12345)

    return train_datagen


def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape,
                        weights='imagenet',
                        include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=100,  activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mae'])
    return model


def train_model(model, train_data, test_data, epochs=9, batch_size=16,
                steps_per_epoch=None, validation_steps=None):
    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True,
              loss=tf.keras.losses.MeanAbsoluteError())

    return model
