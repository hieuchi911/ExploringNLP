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
        shape = 3*input_shape[0][-1]
        self.w = self.add_weight(name = "Similarity_weight",
                                 shape = (1, shape),
                                 initializer = "random_normal",
                                 trainable = True)
        super(Similarity, self).build(input_shape)
    
    def call(self, inputs): # inputs is a dictionary comprises of 2 matrices coming from BiLSTM Context and BiLSTM Query, each of size (2n, max_context_length) and (2n, max_query_length)
        h = inputs[0] # Context TENSOR H with shape (?, max_context_length, features)
        u = inputs[1] # Query TENSOR U with shape (?, max_query_length, features)
        
        # Initiate a 1x6d trainable weight vector with random weights. The shape is 1x6d since this vector will be used in multiplication with concatenated version of outputs from Context (H) and Query (U) biLSTMs: S = alpha(H, U)
        # S will be the similarity matrix, is expected to have shape (num_context_words, num_query_words)
        
        #count = 0 # This variable is used to count the number of Context words
        def find_similarity(inputs):
            h, u = inputs
            count = 0
            for i in range(h.get_shape()[0]): # Index of i is the corresponding index of the word in the context
                m = 0
                for j in range(u.get_shape()[0]): # Index of j is the corresponding index of the query word
                    # i and j are of size (200,), transposing them to make them row vectors, then concatenate them with the element-wise multiplication of themselves to earn temp
                    temp = tf.keras.layers.concatenate([h[i], u[j], tf.math.multiply(h[i], u[j])], 0) # temp shape is (600,) so we have to expand it to (600, 1) -> Use expand_dims
                    temp = tf.keras.backend.expand_dims(temp, 1)
                    # self.w1 is of shape (1, 6d) and temp is of shape (6d, 1) -> alpha is of shape (1, 1)
                    alpha = tf.keras.backend.dot(self.w, temp) # The dot product returns a scalar representing similarity between the "words" i and j ( i and j arent words, but they decodes words)
                    if m == 0:
                        i_to_j_relateness = alpha  # This is a temporary array that stores a vector of scalars denoting similarity between all query words and one word i
                    else:
                        i_to_j_relateness = tf.keras.layers.concatenate([i_to_j_relateness, alpha], 1) # add the scalars alpha in, the loop ends and results in i_to_j_relateness being the similarity matrix between the word i and all the words in the query
                    m +=1
                count += 1
                if count == 1:
                    S = i_to_j_relateness
                else:
                    S = tf.keras.layers.concatenate([S, i_to_j_relateness], 0)
                i_to_j_relateness = 0.0
                m = 0
            print("\nS is: ", S)
            return S
        similarity = tf.keras.backend.map_fn(find_similarity, (h, u), dtype = 'float32')
        print("\nsimilarity matrix is: ", similarity)
        print("Trainable weight matrix is: ", self.trainable_weights)
        return  similarity  # S is the similarity of size (max_context_length, max_query_length)
    

if __name__ == "__main__":
    h = np.array([[[1.2,2.0], [2.2,3.3], [35.1,36.1], [24.1,2.5], [5.1,6.1], [1.2,13.1]]])
    u = np.array([[[12.1,13.1], [23.1,2.4], [2.5,2.6], [35.1,3.6]]])
    inputs = {}
    inputs["Context"] = tf.convert_to_tensor(h, dtype = tf.float32)
    inputs["Query"] = tf.convert_to_tensor(u, dtype = tf.float32)
    similarity_layer = Similarity()
    tensor = similarity_layer(inputs)
    print("End of file")