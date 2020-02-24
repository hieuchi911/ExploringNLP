# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:11:16 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Context2Query(layers.Layer):
    def __init__(self, **kwargs):
        super(Context2Query, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Context2Query, self).build(input_shape)
    
    def call(self, u, s): # Context2Query takes as inputs the biLSTM embeddings u of size (1, max_query_length, features = 2n) and the similarity matrix of size (max_context_length, max_query_length) from the similarity layer
        A = tf.keras.activations.softmax(s, axis = 1) # A, of size (num_context_words, num_query_words), is the distribution of similarities between of words in context and in queries
        c2q = []
        m = 0
        for i in A: # i is of shape (1, num_query_words)
            for j in i: # j is a scalar
                if m == 0:
                    sum_of_weighted_query = tf.math.scalar_mul(j, u[0][m]) # sum_of_weighted_queryTrain is of shape (1, 2d = 200)
                else:
                    sum_of_weighted_query += tf.math.scalar_mul(j, u[0][m])
                m +=1
            m = 0
            c2q.append(sum_of_weighted_query) # c2q is expected to be of shape (200, num_context_words) -> need to take transpose
        c2q = tf.convert_to_tensor(np.array(c2q).T) # c2q is now a tensor of size (200, T), encapsulates the RELEVANCE of each Query word to each Context word
        return c2q