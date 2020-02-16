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
                self.np_Context.append(np.transpose(inputContext, (1, 0)))
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
                    self.np_Query.append(np.transpose(inputQuery, (1, 0)))
                    
                    inputQuery = [] # After appending one question matrix, clear it then append the next query matrices
                self.np_Query.append(np.array([0]))     # This indicates the end of the questions related to the context 
                
            #np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
            #self.np_Context = np.transpose(np_Context, (0, 2, 1))
    def step(self, context_train, query_train):


        # Define inputs embeddings:
        
        # Input is for context and query: each with varying number of timesteps, where a row vector of 
        context_inputs = tf.keras.Input(shape = (len(context_train.T), self.settings["n"]), name = "context_inputs") # Each input is a matrix. And in shape, None indicates the number of timesteps because the number of words in each context is different, while settings["n"] is the size of each word embedding, and batch_size, defined later will be the number of words for each context
        query_inputs = tf.keras.Input(shape = (len(query_train.T), self.settings["n"]), name = "query_inputs") # Each input is a matrix. And in shape, None indicates the number of timesteps, while settings["n"] is the size of each word embedding, and batch_size, defined later will be the number of words for each context
        
        in_h = tf.convert_to_tensor(context_train, name = "context_inputs")
        in_u = tf.convert_to_tensor(query_train, name = "query_inputs")
        print(in_h.get_shape())
        print(in_u.get_shape())
        
        
        # LSTM layer for Context and for Query:
        lstm_query = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM layer for query matrix of size dxT
        bidirectional_query_layer = tf.keras.layers.Bidirectional(lstm_query) # Define BiLSTM LAYER by wrapping the lstm_query LAYER
        bidirectional_query_layer_tensor = bidirectional_query_layer(query_inputs) # query TENSOR of size 2dxJ is returned by plugging an Input tensor query_inputs
        
        
        lstm_context = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for context matrix of size dxT
        bidirectional_context_layer = tf.keras.layers.Bidirectional(lstm_context) # Define BiLSTM LAYER by wrapping the lstm_context LAYER
        bidirectional_context_layer_tensor = bidirectional_context_layer(context_inputs) # context TENSOR of size 2dxT is returned by plugging an Input tensor context_inputs
        
        # Initiate a 1x6d trainable weight vector with random weights. The shape is 1x6d since this vector will be used in multiplication with concatenated version of outputs from Context (H) and Query (U) biLSTMs: S = alpha(H, U)
        h = tf.transpose(bidirectional_context_layer_tensor, perm = [1, 2, 0]) # Context TENSOR H
        u = tf.transpose(bidirectional_query_layer_tensor, perm = [1, 2, 0]) # Query TENSOR U
        
        print(h)
        print(u)
        
        S = []
        i_to_j_relateness = []
        for i in h:
            print(i)
            for j in u:
                print(j)
                temp = tf.transpose(tf.concat((h, u, tf.linalg.matmul(h, u))))
                alpha = tf.keras.layers.dot(self.w1.read_value(), temp)
                i_to_j_relateness.append(alpha)
            S.append(i_to_j_relateness)
            i_to_j_relateness = []
        print(np.shape(S))
        #S = np.array(S).transpose()
        
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
                
    
"""
x = tf.keras.layers.concatenate([bidirectional_context_layer_tensor, bidirectional_query_layer_tensor])

model = tf.keras.Model(inputs = [context_inputs, query_inputs], outputs = x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.fit({"context_inputs": np.transpose(np_Context, (0, 2, 1)), "query_inputs": np.transpose(np_Query[1], (0, 2, 1))}, epochs = 3)
#S = 
"""