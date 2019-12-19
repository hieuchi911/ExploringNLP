# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:12:36 2019

@author: Admin
"""

import math
import matplotlib.pyplot as plt
import numpy as np

sine_wave = np.array([math.sin(x) for x in np.arange(200)])

plt.plot(sine_wave[:50])

# We now devide sine_wave into 2 separated sets: training set and testing set with ratio of 3:1 (150:50)
#       TRAINING SET: pairs of in-output (X as input, Y as the 51st "correct" output) are fed into the network
# We will develop a loop going from 0 to 149, as i increment, a new pair of in-output is created

X = []
Y = []

seqLength = 50
for i in range(100):
    X.append(sine_wave[i:i+seqLength])  # This is appended the input layer, an array of 50 values from sine_wave
    Y.append(sine_wave[i+seqLength])    # This is just the 51th value from the previous 50 values of X[i]

X = np.array(X)
X = np.expand_dims(X, axis = 2) # Expand a third dimension, axis number 2

Y = np.array(Y)
Y = np.expand_dims(Y, axis = 1) # Expand a second dimenstion, axis number 1

#       TESTING SET: quite the same to the above, but take values from 150 to 199 in sine_wave
X_test = []
Y_test = []

for i in range(100, 200):
    X_test.append(sine_wave[i:i+seqLength])
    Y_test.append(sine_wave[i+seqLength])

X_test = np.array(X)
X_test = np.expand_dims(X_test, axis = 2)

Y_test = np.array(Y)
Y_test = np.expand_dims(Y_test, axis = 1)

