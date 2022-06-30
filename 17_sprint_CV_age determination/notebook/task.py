from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
import numpy as np


def load_train(path):
	train_datagen = ImageDataGenerator(rescale=1. / 255,
	                                   horizontal_flip=True,
	                                   vertical_flip=True)
	train_datagen_flow = train_datagen.flow_from_directory(
		path,
		target_size=(150, 150),
		batch_size=16,
		class_mode='sparse',
		subset='training',
		seed=12345)
	return (train_datagen_flow)


def create_model(input_shape):
	model = Sequential()
	optimizer = Adam(lr=0.005)

	model.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=input_shape))
	model.add(AvgPool2D(pool_size=(2, 2)))
	model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
	model.add(AvgPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=64, activation='relu'))
	model.add(Dense(units=32, activation='relu'))
	model.add(Dense(units=12, activation='softmax'))

	model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

	return model


def train_model(model, train_data, test_data, batch_size=None,
                steps_per_epoch=None, validation_steps=None,
                verbose=2, epochs=20):
	model.fit(train_data,
	          validation_data=test_data,
	          batch_size=batch_size,
	          steps_per_epoch=steps_per_epoch,
	          validation_steps=validation_steps,
	          verbose=verbose, epochs=epochs)
	return model
