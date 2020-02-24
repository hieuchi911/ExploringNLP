 # -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 09:43:59 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Similarity(layers.Layer):
    def __init__(self, **kwargs):
        super(Similarity, self).__init__(**kwargs)
    
    def build(self, input_shape):
        shape = 3*input_shape["Context"][-1]
        print(type(shape))
        self.w = self.add_weight(name = "Similarity_weight",
                                 shape = (1, shape),
                                 initializer = "random_normal",
                                 trainable = True)
        super(Similarity, self).build(input_shape)
    
    def call(self, inputs): # inputs is a dictionary comprises of 2 matrices coming from BiLSTM Context and BiLSTM Query, each of size (2n, max_context_length) and (2n, max_query_length)
        h = inputs["Context"] # Context TENSOR H with shape (1, max_context_length, features)
        u = inputs["Query"] # Query TENSOR U with shape (1, max_query_length, features)
        S = []
        i_to_j_relatedness = []
        for i in h[0]: # Index of i is the corresponding index of the word in the context
            for j in u[0]: # Index of j is the corresponding index of the query word
                # i and j are of size (features = 200,), transposing them to make them row vectors, then concatenate them with the element-wise multiplication of themselves to earn temp
                temp = tf.concat((i, j, tf.math.multiply(i, j)), 0) # temp shape is (600,) so we have to expand it to (600, 1) -> Use expand_dims
                temp = tf.expand_dims(temp, 1)
                # self.w1 is of shape (1, 6d) and temp is of shape (6d, 1) -> alpha is of shape (1, 1)
                alpha = tf.tensordot(self.w, temp, 1) # The dot product returns a scalar representing similarity between the "words" i and j ( i and j arent words, but they decodes words)
                i_to_j_relatedness.append(float(alpha[0][0])) # add the scalars alpha in, the loop ends and results in i_to_j_relateness being the similarity matrix between the word i and all the words in the query
            S.append(i_to_j_relatedness)
            i_to_j_relatedness = []
        S = np.array(S) # Turn S from a list to an ndarray of size (max_context_length, max_query_length)
        
        return tf.convert_to_tensor(S, dtype=tf.float32)    # S is the similarity of size (max_context_length, max_query_length)
    
 
if __name__ == "__main__":
    h = np.array([[[1.2,2.0], [2.2,3.3], [35.1,36.1], [24.1,2.5], [5.1,6.1], [1.2,13.1]]])
    u = np.array([[[12.1,13.1], [23.1,2.4], [2.5,2.6], [35.1,3.6]]])
    inputs = {}
    inputs["Context"] = tf.convert_to_tensor(h, dtype = tf.float32)
    inputs["Query"] = tf.convert_to_tensor(u, dtype = tf.float32)
    similarity_layer = Similarity()
    tensor = similarity_layer(inputs)
    print("End of file")