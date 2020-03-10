# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:34:14 2020

@author: Admin
path to train file: "/content/drive/My Drive/QaSModel/Question Answering System/Training and testing data/light-training-data.json"

"""

import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class word2vec():
    def __init__(self, settings, corpus):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        self.vocabulary = []
        self.cleanToken = []
        self.corpus = corpus
        self.tokenizedCorpus = []
        self.w1 = 0
        self.w2 = 0
        self.loss = 0
        self.wordIndex = {}
        self.indexWord = {}
    
    
    def tokenizeCorpus(self):
        self.tokenizedCorpus = word_tokenize(self.corpus)
    
    def buildVocabulary(self):
        # Building vocabulary from the corpus
        weirdlist = [".", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "''", "``", ")", "("]
        print("Building vocabulary\n\n")
        for word in self.tokenizedCorpus:
            if word not in self.vocabulary:
                if word not in weirdlist:
                    self.vocabulary.append(word)
                else:
                    continue
            else:
                continue
        sentence = []
        for word in self.tokenizedCorpus:
            # The dot "." is used to indicate end of a sentence, which will stop the generator from continuing generating words
            if word == "." or word == "?":
                self.cleanToken.append(sentence)
                sentence = []
            elif word not in weirdlist:
                sentence.append(word)
        self.buildIndexFromWord()
        self.buildWordFromIndex()
        
        
    def generate_training_data(self):
        #print("In function generate_training_data\n")
        # Find unique word counts using dictionary
        training_data = []
        #print("In for loop of function generate_training_data\n")
        #for w in tokenizedCorpus:
        for sentence in self.cleanToken:
            for i, word in enumerate(sentence):
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j>=0 and j<len(sentence):
                        training_data.append([self.word2onehot(word), self.word2onehot(sentence[j])])
        
        return np.array(training_data)
       
    def buildIndexFromWord(self):
        m = 0
        for i in self.vocabulary:
            self.wordIndex[i] = m
            m += 1    
                
    def buildWordFromIndex(self):
        for j in range(len(self.vocabulary)):
            self.indexWord[j] = self.vocabulary[j]
    
    def getIndexFromWord(self, word):
        return self.wordIndex[word]
        
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, len(self.vocabulary))]
        word_vec[self.wordIndex[word]] = 1
        return word_vec
    
    def train(self, training_data):
        
        input_layer = tf.keras.Inputs(shape = (len(self.vocabulary)), dtype = "float32", name = "input_words")
        rand = tf.random_uniform_initializer(-1.0, 1.0)
        hidden_layer = tf.keras.layers.Dense(self.n, input_shape = (len(self.vocabulary),), use_bias = True, kernel_initializer = rand)
        hidden_tensor = hidden_layer((input_layer))
        output_layer = tf.keras.layers.Dense(len(self.vocabulary), activation = 'softmax', use_bias = True, kernel_initializer = rand, name = "output_layer")(hidden_tensor)

        
        w2v_model = tf.keras.Model(input_layer, output_layer)
        adam = tf.keras.optimizers.Adam(learning_rate = self.lr)
        w2v_model.compile(loss = loss_funct, optimizer = adam, metrics = ['accuracy'])
        
        
        w2v_model.fit({"input_words": training_data[0]}, {"output_layer": training_data[1]}, epochs = 5, batch_size = 64)
        
        self.w1 = hidden_layer.get_weights()[0]
    
    def vec_sim(self, word, top_n):
        v_w1 = self.w1[self.wordIndex[word]] # embedding for the word 'word' is a row vector from matrix w1. Matrix w1 has the shape of (num_words, n), with n predefined in settings
        print("\n___In vec_sim, v_w1 shape is: ", np.shape(v_w1))
        word_sim = {}
        for m in range(len(self.vocabulary)):
            if self.vocabulary[m] != word:
                v_w2 = self.w1[self.wordIndex[self.vocabulary[m]]]
                v_w1a = v_w1.reshape(1, self.n)
                v_w2a = v_w2.reshape(1, self.n)
                cos_lib = cosine_similarity(v_w1a, v_w2a) 
                theWord = self.vocabulary[m]
                word_sim[theWord] = cos_lib
            else:
                cos_lib = cosine_similarity(v_w1a, v_w2a) 
                continue
        words_sorted = sorted(word_sim.items(), key = lambda kv: kv[1], reverse=True)
        print("Original word: " + word)
        for word1, sim in words_sorted[:top_n]:
            print(word1, sim)

def loss_funct(self, y_pred, y_true):
    def calculate_loss(inputs):
        true, pred = inputs
        loss = tf.keras.backend.sum(-tf.keras.backend.log(pred)*true)
        return loss
    loss = tf.keras.backend.map_fn(calculate_loss, (y_pred, y_true), dtype = tf.float32)
    return tf.keras.backend.mean(loss, axis = 0)


if __name__ == "__main__":
    file = open("/content/drive/My Drive/QaSModel/Word2Vec/testdoc.txt", "r")
    corpus = file.read()
    print("Finish reading\n")
    settings = {
	    'window_size': 2,      	# context window +- center word
	    'n': 100,	         	# dimensions of word embeddings, also refer to size of hidden layer
	    'epochs': 1,	     	# number of training epochs
	    'learning_rate': 0.001	# learning rate
    }


    word2vecAlg = word2vec(settings, corpus)
    word2vecAlg.tokenizeCorpus()
    word2vecAlg.buildVocabulary()
    trainingData = word2vecAlg.generate_training_data()
    word2vecAlg.train(trainingData)