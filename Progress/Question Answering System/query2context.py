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
        z = []  # z, of size (1, num_context_words) is a vector whose elements are each max of the corresponding row in similarity matrix S. z encapsulates the Query word that is most
                # relevant to each  Context word (here, only Context words that are really relevant to a query will be shown in z by their high similarity value, and for 
                # Context words that cannot pay tribute to the answer, they will be neglect and assigned values close to zero)
        for i in s:
            z.append(np.amax(i))
        b = softmax(z) # apply softmax on all elements of z and store in b. b is of shape (1, num_context_words)
        m = 0
        q2c = []
        for scalar in b:
            if m == 0:
                sum_of_weighted_context = tf.math.scalar_mul(scalar, h[0][m])   # Scalar is achieved from b (or z), if it is low, then when being multiplied with 
                                                                                # each (1, 200) vector of h (this vector corresponds to the contextual representation
                                                                                # of that Context word), the scalar causes a decrease in that word's vector, hence decrease
                                                                                # its contribution to the sum_of_weighted_context. But if it is high, then the corresponding vector
                                                                                # being multiplied with will pay much contribution to sum_of_weighted_context. Then, the accumulated vector
                                                                                # sum_of_weighted_context now represents important Context words that answer the Query. Being duplicated
                                                                                # to form q2c, q2c will now decodes the information about most important Context words.
            else:
                sum_of_weighted_context += tf.math.scalar_mul(scalar, h[0][m])
            m += 1
        for scalar in b:
            q2c.append(sum_of_weighted_context) # duplicate the row vector sum_of_weighted_context of shape (1, 200) for num_words_query times
        q2c = tf.convert_to_tensor(np.array(q2c).T) # q2c is of size (200, T), encapsulates information about the most important words in the Context w.r.t the Query
        return q2c