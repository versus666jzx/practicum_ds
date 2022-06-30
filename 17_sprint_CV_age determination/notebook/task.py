from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
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

    train_datagen = datagen.flow_from_directory(
        path,
        batch_size=16,
        target_size=(150, 150),
        subset='training',
        class_mode='sparse',
        seed=12345)
    return train_datagen


def create_model(input_shape):
    optimizer = Adam(lr=0.03)
    backbone = ResNet50(input_shape=(150, 150, 3),
                        weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                        include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(Dense(units=12))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    return model


def train_model(model, train_data, test_data, epochs=5, batch_size=16,
                steps_per_epoch=None, validation_steps=None):
    model.fit(train_data,
              validation_data=test_data,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model
