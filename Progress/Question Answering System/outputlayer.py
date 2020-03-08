# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:43:25 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class OutputLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        shape_wp1 = 5*input_shapes[1][1]
        shape_wp2 = shape_wp1
        self.wp1 = self.add_weight(name = "Start_index_pred",
                                   shape = (1, shape_wp1),
                                   initializer = "random_normal",
                                   trainable = True)
        self.wp2 = self.add_weight(name = "Start_index_pred",
                                   shape = (1, shape_wp2),
                                   initializer = "random_normal",
                                   trainable = True)
        super(OutputLayer, self).build(input_shapes)
    
    def call(self, inputs): # inputs comprises of: G(?, 800, 114) m1_tensor(?, 200, 114) and m2_tensor(?, 200, 114)
        G, m1_tensor, m2_tensor = inputs
        
        def concat_G(my_inputs):
            G, m_tensor = my_inputs
            result = tf.keras.backend.transpose(tf.keras.layers.concatenate([G, m_tensor], 0))  # result was of shape (10d, T) then transposed to (T, 10d)
            return result
        
        def output_layer(inputs_Gs):
            G_M1 = inputs_Gs
            m = 0
            for i in range(0, G_M1.get_shape()[0]): # i is of shape (10d, )
                scalar = tf.keras.backend.dot(self.wp1, tf.keras.backend.expand_dims(G_M1[i], 1))
                if m == 0:
                    p1 = scalar
                else:
                    p1 = tf.keras.layers.concatenate([p1, scalar], 1)
                m += 1
            p1 = tf.keras.backend.reshape(p1, [114])
            print("\np1 is: ", p1)
            return p1
            
        def output_layer1(input_G):
            G_M2 = input_G    
            m = 0
            for i in range(0, G_M2.get_shape()[0]):
                scalar = tf.keras.backend.dot(self.wp2, tf.keras.backend.expand_dims(G_M2[i], 1))
                if m == 0:
                    p2 = scalar
                else:
                    p2 = tf.keras.layers.concatenate([p2, scalar], 1)
                m += 1
            p2 = tf.keras.backend.reshape(p2, [114])
            print("\np2 is: ", p2)
            print("\nFinished 2 for loops!!!!!!!!!!!!\n")
            return p2
   
        G_M1 = tf.keras.backend.map_fn(concat_G, (G, m1_tensor), dtype = 'float32')
        print("\nG_M1 is: ", G_M1)
        
        G_M2 = tf.keras.backend.map_fn(concat_G, (G, m2_tensor), dtype = 'float32')
        print("G_M2 is: ", G_M2)

        p1 = tf.keras.backend.map_fn(output_layer, (G_M1), dtype = 'float32')
        p2 = tf.keras.backend.map_fn(output_layer1, (G_M2), dtype = 'float32')
        

        print("\nOUT!!!\np1 is: ", p1)
        print("p2 is: ", p2)
        
        p1_pred = tf.keras.activations.softmax(p1)

        p2_pred = tf.keras.activations.softmax(p2)
        
        output_final = tf.keras.backend.stack([p1_pred, p2_pred], axis = 1)
        
        print("\nOutput_final is: ", output_final)
        print("Finished forming prediction vectors: " + str(output_final.get_shape()) + "\n")
        print("Trainable weight matrix is: ", self.trainable_weights)
        return output_final
    
if __name__ == "__main__":
    G_M1 = np.array([[1.2,2.0], [2.2,3.3], [35.1,36.1], [24.1,2.5], [5.1,6.1], [1.2,13.1]])
    G_M2 = np.array([[12.1,13.1], [23.1,2.4], [2.5,2.6], [35.1,3.6], [35.1,3.6], [35.1,3.6]])
    inputs = {}
    inputs["G_M1"] = tf.convert_to_tensor(G_M1, dtype = tf.float32)
    inputs["G_M2"] = tf.convert_to_tensor(G_M2, dtype = tf.float32)
    output_layer = OutputLayer()
    hi = output_layer(inputs)
    print(hi)
    print("End of file")
        