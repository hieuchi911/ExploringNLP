# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:07:35 2019

@author: Admin
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_lables) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scaling pixels down to deal with them at ease
train_images = train_images/255.0
test_images = test_images/255.0

# Modelling neural network with input of 28x28 values representing all pixels of each image, a hidden layer with 50 neurons implementing Rectified Linear Unit and an output layer 
# with 10 neurons, each of which represents the probability that the given input (an image) is of that label, implementing softmax to get probabilities summing up to 1
model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(50, activation ="relu"),
        keras.layers.Dense(10, activation = "softmax")
        ])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
# Training the model
model.fit(train_images, train_labels, epochs = 1)

# Make prediction for all test images
prediction = model.predict(test_images)
# Print out some comparisons between real image - real label and predicted label
for i in range(6, 11):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_lables[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
