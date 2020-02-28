# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:35:18 2020

@author: Admin
"""

import tensorflow as tf
from tensorflow.keras import layers



class MegaMerge(layers.Layer):
    def __init__(self, **kwargs):
        super(MegaMerge, self).__init__(**kwargs)
    
    def build(self, input_shapes):
        super(MegaMerge, self).build(input_shapes)
    
    def call(self, h, c2q, q2c, max_context_length):
        # At this point, H is of shape(1, num_context_words, 200); C2Q and Q2C are of shape(200, num_context_words)
        G = 0.0
        
        for c in range(0, max_context_length):
            # beta below is a row vector of size (1, 800)
            beta = tf.concat((tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(c2q)[c], 1), tf.math.multiply(tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(c2q)[c], 1)), tf.math.multiply(tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(q2c)[c], 1))), 0) # 0 here indicates that concatenation is done along rows, since each element in the concatenate func is of size (200,)
            if c == 0:
                G = beta                    # G, after being transposed to be of size (800, num_context_words), is the QUERY-AWARE REPRESENTATION of ALL CONTEXT WORDS, where each column
                                            # vector is a concatenation of the Context word itself, the query-context-relevance subvector and the importance-to-query
                                            # subvector. Each column of G is the representation of a Context word that is aware of the existence of the Query and
                                            # has incorporated-and-relevant information from the Query.
            else:
                G = tf.concat((G, beta), 1)
        print("G is: ", G)
        print("Finished megamerging H, C2Q and Q2C to G: " + str(G.get_shape()) + "\n")
        return G