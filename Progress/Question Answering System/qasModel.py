# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:34:14 2020

@author: Admin
path to train file: "Training and testing data\\light-training-data.json"

"""

import numpy as np
import tensorflow as tf
import json
from word2Vec import word2vecClass
from nltk.tokenize import word_tokenize
from scipy.special import softmax

class MyQuestionAnsweringModel(tf.keras.Model):
    
    def __init__(self):
        super(MyQuestionAnsweringModel, self).__init__()
        self.settings = {'window_size': 2, 'n': 100, 'epochs': 5, 'learning_rate': 0.001}
        trainable_weight_vector = np.random.random([1, 6*self.settings["n"]]).astype(np.float32)
        self.w1 = tf.Variable(trainable_weight_vector) # trainable weight vector w1
        

    def tokenizeCorpus(self, corpus):
        corpusWithNoPunctuals = []
        tokenized = word_tokenize(corpus)
        weirdlist = [".", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "''", "``", ")", "("]
        for word in tokenized:
            if word not in weirdlist:
                corpusWithNoPunctuals.append(word)
            else:
                continue
        return corpusWithNoPunctuals
    def extract_training_inputs(self, path):
        # Define inputs to the model: word embeddings for Context and word embeddings for Query
        with open(path) as f:
            data = json.loads(f.read())
        inputContext = []
        ContextCorpus = "" # This contains all contexts in training file
        inputQuery = []
        QueryCorpus = "" # This contains all queries in training file
        
        
        # Now create data to pass as input to the first BiLSTM layer of our model
        # Concatenate all contexts to create word2vec vector:
        """ UNCOMMENT THIS TO RETREIVE OFFICIAL CODE
        for topic in data["data"]:
            for context in topic["paragraphs"]:
                ContextCorpus += " " + context["context"]
                for query in context["qas"]:
                    QueryCorpus += " " + query["question"]
        wordEmbeddingContext = word2vecClass(settings, ContextCorpus)
        wordEmbeddingContext.tokenizeCorpus()
        wordEmbeddingContext.buildVocabulary()
        wordEmbeddingContext.train(wordEmbeddingContext.generate_training_data())
        # wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        wordEmbeddingQuery = word2vecClass(settings, QueryCorpus)
        wordEmbeddingQuery.tokenizeCorpus()
        wordEmbeddingQuery.buildVocabulary()
        wordEmbeddingQuery.train(wordEmbeddingQuery.generate_training_data())
        # wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        training_context_data = []
        training_query_data = []
        
        for topic in data["data"]:
            for context in data["data"][0]["paragraphs"]:
                # Create word2vec embedding for a Context
                Context = tokenizeCorpus(context["context"])        
                for word in Context:
                    # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                    # Also, w1 shape is (num_words, d = n) so we have to take transpose of w1, then convert np.ndarray to list, so we can subsequently convert the list to tensor
                    inputContext.append(wordEmbeddingContext.w1[wordEmbeddingContext.wordIndex[word]].T.tolist())
                training_context_data.append(inputContext)
                # For each of queries about given Context, create word2vec embedding for that query
                for query in context["qas"]:
                    Query = tokenizeCorpus(query["question"])
                    for word in Query:
                        # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                        # Also, w1 shape is (num_words, d = n) so we have to take transpose of w1, then convert np.ndarray to list, so we can subsequently convert the list to tensor
                        inputQuery.append(wordEmbeddingQuery.w1[wordEmbeddingQuery.wordIndex[word]].T.tolist())
                    training_query_data.append(inputQuery)
                np_Query = np.asarray(training_query_data, dtype = np.float32) # inputQuery here is the query input training_data that will be passed to the fit function of Keras Model
            np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
        """
        a = 0
        for context in data["data"][0]["paragraphs"][0:2]:
            if a==1:
                break
            ContextCorpus += " " + context["context"]
            for query in context["qas"]:
                QueryCorpus += " " + query["question"]
                if a == 3:
                    break
                else:
                    continue
                a+=1
        print("_______________________CONTEXTS__________________________ \n\n" + ContextCorpus)
        print(type(ContextCorpus))
        
        print("_______________________QUERIES__________________________ \n\n" + QueryCorpus)
        print(type(QueryCorpus))
        
        wordEmbeddingContext = word2vecClass(self.settings, ContextCorpus)
        wordEmbeddingContext.tokenizeCorpus()
        wordEmbeddingContext.buildVocabulary()
        wordEmbeddingContext.train(wordEmbeddingContext.generate_training_data())
        # wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        wordEmbeddingQuery = word2vecClass(self.settings, QueryCorpus)
        wordEmbeddingQuery.tokenizeCorpus()
        wordEmbeddingQuery.buildVocabulary()
        wordEmbeddingQuery.train(wordEmbeddingQuery.generate_training_data())
        # wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        #training_context_data = []
        #training_query_data = []
        self.np_Context = []
        self.np_Query = []
        for i in range(1):
            for context in data["data"][0]["paragraphs"][0:1]:
                
                # Create word2vec embedding for a Context
                Context = self.tokenizeCorpus(context["context"])        
                for word in Context:
                    # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                    # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                    inputContext.append(wordEmbeddingContext.w1[wordEmbeddingContext.getIndexFromWord(word)].tolist())
                inputContext = np.asarray(inputContext, dtype = np.float32)
                self.np_Context.append(inputContext)
                inputContext = [] # After appending one context matrix, clear it then append the next context matrices
                print(context["context"])
                # For each of queries about given Context, create word2vec embedding for that query
                for query in context["qas"]:
                    Query = self.tokenizeCorpus(query["question"])
                    for word in Query:
                        # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                        # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                        inputQuery.append(wordEmbeddingQuery.w1[wordEmbeddingQuery.wordIndex[word]].tolist()) # shape of inputQuery is (num_words, n)
                    inputQuery = np.asarray(inputQuery, dtype = np.float32)
                    self.np_Query.append(inputQuery)
                    
                    inputQuery = [] # After appending one question matrix, clear it then append the next query matrices
                self.np_Query.append(np.array([0]))     # This indicates the end of the questions related to the context 
                
            #np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
            #self.np_Context = np.transpose(np_Context, (0, 2, 1))
    
    def step(self, context_train, query_train): # Each of the argument here has a shape of (num_words, n), num_words will be equal to the number of time steps, n will be equal to num_features in the input of LSTM


        #_____________________________ DEFINE INPUTS EMBEDDINGS USING BILSTMs _____________________________
        
        # Context and query inputs to LSTM layers must each be a 3D tensor: 3D are 1 being batch size, num_words being the number of timesteps, 200 being number of features
        in_h = tf.convert_to_tensor(context_train, name = "context_inputs")
        in_h = tf.expand_dims(in_h, 0)  # Expand one more dimension to plug in the LSTM layer, which corresponds to the batch_size
        print(in_h.get_shape())
        in_u = tf.convert_to_tensor(query_train, name = "query_inputs")
        in_u = tf.expand_dims(in_u, 0)  # Expand one more dimension to plug in the LSTM layer, which corresponds to the batch_size
        print(in_u.get_shape())
        
        # LSTM layer for Context and for Query:
        lstm_context = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for context matrix of size dxT
        bidirectional_context_layer = tf.keras.layers.Bidirectional(lstm_context) # Define BiLSTM LAYER by wrapping the lstm_context LAYER
        bidirectional_context_layer_tensor = bidirectional_context_layer(in_h) # context TENSOR of size 2dxT is returned by plugging an Input tensor context_inputs
        
        lstm_query = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM layer for query matrix of size dxT
        bidirectional_query_layer = tf.keras.layers.Bidirectional(lstm_query) # Define BiLSTM LAYER by wrapping the lstm_query LAYER
        bidirectional_query_layer_tensor = bidirectional_query_layer(in_u) # query TENSOR of size 2dxJ is returned by plugging an Input tensor query_inputs
        #
        #
        #_____________________________ FORMING SIMILARITY MATRIX S _____________________________
        
        # Initiate a 1x6d trainable weight vector with random weights. The shape is 1x6d since this vector will be used in multiplication with concatenated version of outputs from Context (H) and Query (U) biLSTMs: S = alpha(H, U)
        h = bidirectional_context_layer_tensor # Context TENSOR H with shape (1, num_words, features)
        u = bidirectional_query_layer_tensor # Query TENSOR U with shape (1, num_words, features)
        
        S = [] # S will be the similarity matrix, is expected to have shape (num_context_words, num_query_words)
        i_to_j_relateness = []  # This is a temporary array that stores a vector of scalars denoting similarity between all query words and one word i
                
        for i in h[0]: # Index of i is the corresponding index of the word in the context
            for j in u[0]: # Index of j is the corresponding index of the word in the query
                # i and j are of size (200,), transposing them to make them row vectors, then concatenate them with the element-wise multiplication of themselves to earn temp
                temp = tf.concat((tf.transpose(i), tf.transpose(j), tf.math.multiply(tf.transpose(i), tf.transpose(j))), 0) # temp shape is (600,) so we have to expand it to (1, 600) -> Use expand_dims
                temp = tf.expand_dims(temp, 1)
                
                alpha = tf.tensordot(self.w1.read_value(), temp, 1) # The dot product returns a scalar representing similarity between the "words" i and j ( i and j arent words, but they decodes words)
                i_to_j_relateness.append(float(alpha[0][0])) # add the scalars alpha in, the loop ends and results in i_to_j_relateness being the similarity matrix between the word i and all the words in the query
            S.append(i_to_j_relateness)
            i_to_j_relateness = []
        S = np.array(S) # Turn S from a list to an ndarray of size (num_context_words, num_query_words)
        #
        #
        #_____________________________ FORMING CONTEXT TO QUERY MATRIX FROM S _____________________________
        
        A = tf.keras.activations.softmax(tf.convert_to_tensor(S, dtype=tf.float32), axis = 1) # A, of size (num_context_words, num_query_words), is the distribution of similarities between of words in context and in queries
        c2q = []
        m = 0
        for i in A: # i is of shape (1, num_query_words)
            for j in i: # j is a scalar
                print(u[0][m].get_shape())
                if m == 0:
                    sum_of_weighted_query = tf.math.scalar_mul(j, u[0][m]) # sum_of_weighted_queryTrain is of shape (1, d = 200)
                else:
                    sum_of_weighted_query += tf.math.scalar_mul(j, u[0][m])
                m +=1
            m = 0
            c2q.append(sum_of_weighted_query) # U_Context is expected to be of shape (200, num_context_words)
        c2q = tf.convert_to_tensor(np.array(c2q).T)
        #
        #
        #_____________________________ FORMING QUERY TO CONTEXT MATRIX FROM S _____________________________
        
        z = [] # z is a vector whose elements are each max of the corresponding row in similarity matrix S
        for i in S:
            z.append(np.amax(i))
        b = softmax(z) # apply softmax on all elements of z and store in b. b is of shape (1, num_context_words)
        m = 0
        q2c = []
        for scalar in b:
            if m == 0:
                sum_of_weighted_context = tf.math.scalar_mul(scalar, h[0][m])
            else:
                sum_of_weighted_context += tf.math.scalar_mul(scalar, h[0][m])
            m += 1
        for scalar in b:
            q2c.append(sum_of_weighted_context) # duplicate num_words_query times the row vector of shape (1, 200)
        q2c = tf.convert_to_tensor(np.array(q2c).T)
        #
        #
        #_____________________________ MEGAMERGE MATRICES: H, C2Q AND Q2C _____________________________
        
    """    
    def iterating(self, h_2D, query):
        self.S.append(tf.map_fn(lambda x: self.concatenate(x, h_2D), query))
        
    def concatenate(self, u_2D, h_2D):
        i_to_j_relateness = []
        temp = tf.transpose(tf.concat((h_2D, u_2D, tf.linalg.matmul(h_2D, u_2D))))
        alpha = tf.keras.layers.dot(self.w1.read_value(), temp)
        i_to_j_relateness.append(alpha)
        return i_to_j_relateness
"""
        
if __name__ == "__main__":
    myModel = MyQuestionAnsweringModel()
    myModel.extract_training_inputs("Training and testing data\\light-training-data.json")
    for i in myModel.np_Context:
        for j in myModel.np_Query:
            if np.shape(j) == (1,):
                print("break!!!")
                break
            else:
                myModel.step(i, j)
                break
                
    
"""
x = tf.keras.layers.concatenate([bidirectional_context_layer_tensor, bidirectional_query_layer_tensor])

model = tf.keras.Model(inputs = [context_inputs, query_inputs], outputs = x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.fit({"context_inputs": np.transpose(np_Context, (0, 2, 1)), "query_inputs": np.transpose(np_Query[1], (0, 2, 1))}, epochs = 3)
#S = 
"""