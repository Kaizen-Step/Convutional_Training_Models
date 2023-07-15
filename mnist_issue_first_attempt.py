# Import packages

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


# Data Processing

mnist_datasets, mnist_info = tfds.load(
    name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = mnist_datasets['train'], mnist_datasets['test']


num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# Define scaling function to normalize the inputs images

def scale(image, lable):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, lable


# Map the scaling funtion to input variables

scaled_validation_train_data = mnist_train.map(scale)


# Set the buffer size for each shuffle for better shuffling

buffer_size = 1000


# Shuffle the data

shuffled_vaidation_train_data = scaled_validation_train_data.shuffle(
    buffer_size)


# Define Validation and train Data-Sets

validation_data = shuffled_vaidation_train_data.take(num_validation_samples)
train_data = shuffled_vaidation_train_data.skip(num_validation_samples)


# Define batch size and batching data sets for better computational power

batch_size = 100

train_data = train_data.batch(batch_size)
validation_data = validation_data.batch(num_validation_samples)

validation_inputs, validation_targets = next(iter(validation_data))

# Outline the Model

input_size = 784
output_size = 10
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')

])


# Choose Optimizer and Loss function

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Training Model


num_epochs = 30

model.fit(train_data, epochs=num_epochs, validation_data=(
    validation_inputs, validation_targets), verbose=2)

# Test the Model

test_loss, test_accuracy = model.evaluate(mnist_test)
