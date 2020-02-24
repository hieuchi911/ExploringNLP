# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:43:25 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
from scipy.special import softmax
import numpy as np

class OutputLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        shape_wp1 = input_shapes["G_M1"][-1]
        shape_wp2 = shape_wp1
        print(type(input_shapes["G_M1"]))
        print(shape_wp1)
        self.wp1 = self.add_weight(name = "Start_index_pred",
                                   shape = (1, shape_wp1),
                                   initializer = "random_normal",
                                   trainable = True)
        self.wp2 = self.add_weight(name = "Start_index_pred",
                                   shape = (1, shape_wp2),
                                   initializer = "random_normal",
                                   trainable = True)
        super(OutputLayer, self).build(input_shapes)
    
    def call(self, inputs):
        p1 = []
        p2 = []
        for i in inputs["G_M1"]:
            # i is of shape (10n, )
            scalar = tf.tensordot(self.wp1.read_value(), tf.transpose(i), 1)
            p1.append(float(scalar[0]))
        for i in inputs["G_M2"]:
            scalar = tf.tensordot(self.wp2.read_value(), tf.transpose(i), 1)
            p2.append(float(scalar[0]))
        p1_pred = tf.convert_to_tensor(softmax(p1))
        p2_pred = tf.convert_to_tensor(softmax(p2))
        return p1_pred, p2_pred
    
if __name__ == "__main__":
    G_M1 = np.array([[1.2,2.0], [2.2,3.3], [35.1,36.1], [24.1,2.5], [5.1,6.1], [1.2,13.1]])
    G_M2 = np.array([[12.1,13.1], [23.1,2.4], [2.5,2.6], [35.1,3.6]])
    inputs = {}
    inputs["G_M1"] = tf.convert_to_tensor(G_M1, dtype = tf.float32)
    inputs["G_M2"] = tf.convert_to_tensor(G_M2, dtype = tf.float32)
    output_layer = OutputLayer()
    p1_pred, p2_pred = output_layer(inputs)
    print("End of file")
        