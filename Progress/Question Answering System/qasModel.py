# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:34:14 2020

@author: Admin
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
from word2Vec import word2vecClass
from nltk.tokenize import word_tokenize


def tokenizeCorpus(corpus):
    corpusWithNoPunctuals = []
    tokenized = word_tokenize(corpus)
    weirdlist = [".", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "''", "``", ")", "("]
    for word in tokenized:
        if word not in weirdlist:
            corpusWithNoPunctuals.append(word)
        else:
            continue
    return corpusWithNoPunctuals
# Define inputs to the model: word embeddings for Context and word embeddings for Query
with open("Training and testing data\\light-training-data.json") as f:
    data = json.loads(f.read())
inputContext = []
ContextCorpus = "" # This contains all contexts in training file
inputQuery = []
QueryCorpus = "" # This contains all queries in training file
settings = {'window_size': 2, 'n': 100, 'epochs': 5, 'learning_rate': 0.001}

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
for context in data["data"][0]["paragraphs"][0:1]:
    print(context["context"])
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
print(ContextCorpus)
print(type(ContextCorpus))

print(QueryCorpus)
print(type(QueryCorpus))

wordEmbeddingContext = word2vecClass(settings, ContextCorpus)
wordEmbeddingContext.tokenizeCorpus()
wordEmbeddingContext.buildVocabulary()
wordEmbeddingContext.train(wordEmbeddingContext.generate_training_data())
print(wordEmbeddingContext.vocabulary)
# wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
word = "became"
index = wordEmbeddingContext.getIndexFromWord(word)
print(index)
wordEmbeddingQuery = word2vecClass(settings, QueryCorpus)
wordEmbeddingQuery.tokenizeCorpus()
wordEmbeddingQuery.buildVocabulary()
wordEmbeddingQuery.train(wordEmbeddingQuery.generate_training_data())
# wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
training_context_data = []
training_query_data = []

for i in range(1):
    i=1
    for context in data["data"][0]["paragraphs"][0:1]:
        
        # Create word2vec embedding for a Context
        Context = tokenizeCorpus(context["context"])        
        for word in Context:
            # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
            # Also, w1 shape is (num_words, d = n) so we have to take transpose of w1, then convert np.ndarray to list, so we can subsequently convert the list to tensor
            inputContext.append(wordEmbeddingContext.w1[wordEmbeddingContext.getIndexFromWord(word)].T)
        training_context_data.append(inputContext)
        print(context["context"])
        if i==3:
            break
        # For each of queries about given Context, create word2vec embedding for that query
        for query in context["qas"]:
            Query = tokenizeCorpus(query["question"])
            for word in Query:
                # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                # Also, w1 shape is (num_words, d = n) so we have to take transpose of w1, then convert np.ndarray to list, so we can subsequently convert the list to tensor
                inputQuery.append(wordEmbeddingQuery.w1[wordEmbeddingQuery.wordIndex[word]].T.tolist())
            training_query_data.append(inputQuery)
            i+=1
            if i==3:
                break
            print(query["question"])
        np_Query = np.asarray(training_query_data, dtype = np.float32) # inputQuery here is the query input training_data that will be passed to the fit function of Keras Model
    np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model

print(np.shape(np_Context))
print(np.shape(np_Query))
# Define inputs embeddings:

# Input is for context and query: each with varying number of timesteps, where a row vector of 
context_inputs = tf.keras.Input(shape = (None, settings["n"],), name = "context_inputs") # Each input is a matrix. And in shape, None indicates the number of timesteps, while settings["n"] is the size of each word embedding, and batch_size, defined later will be the number of words for each context
query_inputs = tf.keras.Input(shape = (None, settings["n"],), name = "query_inputs") # Each input is a matrix. And in shape, None indicates the number of timesteps, while settings["n"] is the size of each word embedding, and batch_size, defined later will be the number of words for each context

# LSTM layer for Context and for Query:
lstm_context = tf.keras.layers.LSTM(settings["n"], activation = "tanh", trainable = False) # Define an LSTM layer for context matrix of size dxT
bidirectional_context = tf.keras.layers.Bidirectional(lstm_context)(context_inputs) # a new context matrix of size 2dxT

lstm_query = tf.keras.layers.LSTM(settings["n"], activation = "tanh", trainable = False) # Define an LSTM layer for query matrix of size dxT
bidirectional_query = tf.keras.layers.Bidirectional(lstm_query)(query_inputs) # a new context matrix of size 2dxT



x = tf.keras.layers.concatenate([bidirectional_context, bidirectional_query])

model = tf.keras.Model(inputs = [context_inputs, query_inputs], outputs = x)
model.compile()
model.fit(np_Context, np_Query, epochs = 3)
#S = 