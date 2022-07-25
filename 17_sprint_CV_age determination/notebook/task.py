import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AvgPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_train(path):
	train_datagen = ImageDataGenerator(
	    horizontal_flip=True,
		vertical_flip=True,
		rascale=1 / 225.
	)

	train_datagen_flow = train_datagen.flow_from_directory(
		path,
		target_size=(150, 150),
		batch_size=16,
		class_mode='raw',
		seed=12345)
	return train_datagen_flow


def create_model(input_shape):
	model = Sequential()
	optimizer = Adam(lr=0.0001)
	model.add(Conv2D(6, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
	model.add(AvgPool2D(pool_size=(2, 2)))
	model.add(Conv2D(16, (3, 3), padding='valid', activation='relu'))
	model.add(AvgPool2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=64, activation='relu'))
	model.add(Dense(units=12, activation='softmax'))

	model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(), metrics=['acc'])

	return model


def train_model(model, train_data, test_data,
				batch_size=32, steps_per_epoch=None,
				validation_steps=None, epochs=2):
	model.fit(train_data,
			  epochs=epochs,
			  validation_data=test_data,
			  batch_size=batch_size,
			  steps_per_epoch=steps_per_epoch,
			  validation_steps=validation_steps
			  )

	return model


def load_train(path):
	labels = pd.read_csv(path + 'labels.csv')
	datagen = ImageDataGenerator(
		validation_split=0.25,
		horizontal_flip=True,
		vertical_flip=True,
		rescale=1. / 255)

	train_gen_flow = datagen.flow_from_dataframe(
		dataframe=labels,
		directory=path + 'final_files/',
		x_col='file_name',
		y_col='real_age',
		target_size=(224, 224),
		batch_size=16,
		class_mode='raw',
		subset='training',
		seed=12345)

	return train_gen_flow


def load_test(path):
	labels = pd.read_csv(path + 'labels.csv')
	datagen = ImageDataGenerator(
		validation_split=0.25,
		rescale=1. / 255)

	test_gen_flow = datagen.flow_from_dataframe(
		dataframe=labels,
		directory=path + 'final_files/',
		x_col='file_name',
		y_col='real_age',
		target_size=(224, 224),
		batch_size=16,
		class_mode='raw',
		subset='validation',
		seed=12345)

	return test_gen_flow


def create_model(input_shape):
	optimizer = Adam(lr=0.0001)
	backbone = ResNet50(input_shape=input_shape,
						weights='imagenet',
						include_top=False)
	model = Sequential()
	model.add(backbone)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(units=1, activation='relu'))
	model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
				  metrics=['mae'])
	return model


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
				steps_per_epoch=None, validation_steps=None):
	model.fit(train_data,
			  validation_data=test_data,
			  batch_size=batch_size, epochs=epochs,
			  steps_per_epoch=steps_per_epoch,
			  validation_steps=validation_steps,
			  verbose=2)

	return model
