# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:35:18 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

max_context_length = 500

class MegaMerge(layers.Layer):
    def __init__(self, **kwargs):
        super(MegaMerge, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        super(MegaMerge, self).build(input_shapes)
    
    def call(self, h, c2q, q2c):
        # At this point, H is of shape(1, num_context_words, 200); C2Q and Q2C are of shape(200, num_context_words)
        G = []
        for c in range(0, max_context_length):
            # beta below will be a row vector of size (1, 800)
            beta = tf.concat((h[0][c], tf.transpose(c2q)[c], tf.math.multiply(h[0][c], tf.transpose(c2q)[c]), tf.math.multiply(h[0][c], tf.transpose(q2c)[c])), 0) # 1 here indicates that concatenation is done along columns
            G.append(tf.transpose(beta))    # G, of size (800, num_context_words), is the QUERY-AWARE REPRESENTATION of ALL CONTEXT WORDS, where each column
                                            # vector is a concatenation of the Context word itself, the query-context-relevance subvector and the importance-to-query
                                            # subvector. Each column of G is the representation of a Context word that is aware of the existence of the Query and
                                            # has incorporated-and-relevant information from the Query.
        G = tf.convert_to_tensor(np.array(G).T)
        return G