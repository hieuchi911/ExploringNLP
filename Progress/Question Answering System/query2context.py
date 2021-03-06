# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:11:16 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from scipy.special import softmax

class Query2Context(layers.Layer):
    def __init__(self, **kwargs):
        super(Query2Context, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Query2Context, self).build(input_shape)
    
    def call(self, h, s): # Query2Context takes as inputs the biLSTM embeddings h of size (1, max_context_length, features = 2n) and the similarity matrix of size (max_context_length, max_query_length) from the similarity layer
                # z, of size (1, num_context_words) is a vector whose elements are each max of the corresponding row in similarity matrix S. z encapsulates the Query word that is most
                # relevant to each  Context word (here, only Context words that are really relevant to a query will be shown in z by their high similarity value, and for 
                # Context words that cannot pay tribute to the answer, they will be neglect and assigned values close to zero)
        
        z1 = tf.keras.backend.max(s, axis = -1) # Take max at each row to generate a column vector z1
        b = tf.keras.layers.Softmax()(tf.keras.backend.expand_dims(z1, 1)) # apply softmax on all elements of z and store in b. b is of shape (1, num_context_words)
        def find_q2c(inputs):
            m = 0
            b, h = inputs
            for scalar in range(0, b.get_shape()[1]):
                if m == 0:
                    sum_of_weighted_context = tf.keras.backend.expand_dims(tf.math.scalar_mul(b[0][scalar], h[m]), 1)
                    # Scalar is achieved from b (or z), if it is low, then when being multiplied with 
                    # each (1, 200) vector of h (this vector corresponds to the contextual representation
                    # of that Context word), the scalar causes a decrease in that word's vector, hence decrease
                    # its contribution to the sum_of_weighted_context. But if it is high, then the corresponding vector
                    # being multiplied with will pay much contribution to sum_of_weighted_context. Then, the accumulated vector
                    # sum_of_weighted_context now represents important Context words that answer the Query. Being duplicated
                    # to form q2c, q2c will now decodes the information about most important Context words.
                else:
                    sum_of_weighted_context += tf.keras.backend.expand_dims(tf.math.scalar_mul(b[0][scalar], h[m]), 1)
                m += 1
            m = 0
            for scalar in range(0, b.get_shape()[1]):
                if m == 0:
                    q2c = sum_of_weighted_context
                else:
                    q2c = tf.keras.layers.concatenate([q2c, sum_of_weighted_context], 1) # duplicate the row vector sum_of_weighted_context of shape (200,) for num_words_query times
                m += 1
                # q2c is of size (200, T), encapsulates information about the most important words in the Context w.r.t the Query
            return q2c
        q2c = tf.keras.backend.map_fn(find_q2c, (b, h), dtype = 'float32')
        print("\nQ2C is: ", q2c)
        print("Trainable weight matrix is: ", self.trainable_weights)
        return q2c