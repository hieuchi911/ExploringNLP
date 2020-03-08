# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:11:16 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers

class Context2Query(layers.Layer):
    def __init__(self, **kwargs):
        super(Context2Query, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Context2Query, self).build(input_shape)
    
    def call(self, u, s): # Context2Query takes as inputs the biLSTM embeddings u of size (1, max_query_length, features = 2n) and the similarity matrix of size (max_context_length, max_query_length) from the similarity layer
        A = tf.keras.layers.Softmax()(s) # A, of size (?, num_context_words, num_query_words), is the distribution of similarities between of words in context and in queries
        
        def find_c2q(inputs):
            m = 0
            count1 = 0
            A, u = inputs
            for i in range(0, A.get_shape()[0]): # i is of shape (1, num_query_words)
                for j in range(0, A.get_shape()[1]): # j is a scalar
                    if m == 0:
                        sum_of_weighted_query = tf.keras.backend.expand_dims(tf.math.scalar_mul(A[i][j], u[m]), 1) # sum_of_weighted_query is of shape (2d = 200)
                    else:
                        sum_of_weighted_query += tf.keras.backend.expand_dims(tf.math.scalar_mul(A[i][j], u[m]), 1)
                    m +=1
                m = 0
                if count1 == 0:
                    c2q = sum_of_weighted_query
                    count1 +=1
                else:
                    c2q = tf.keras.layers.concatenate([c2q, sum_of_weighted_query], 1)
                sum_of_weighted_query = 0.0
            return c2q
        c2q =  tf.keras.backend.map_fn(find_c2q, (A, u), dtype = 'float32') # c2q is now a tensor of size (?, 200, T), encapsulates the RELEVANCE of each Query word to each Context word
        print("\nC2Q is: ", c2q)
        print("Trainable weight matrix is: ", self.trainable_weights)

        return c2q