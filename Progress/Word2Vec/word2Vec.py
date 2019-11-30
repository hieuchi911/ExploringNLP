# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:24:28 2019

@author: Admin
"""

#from gensim
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class word2vecClass():
    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']    
    
    def generate_training_data(self, tokenizedCorpus):
        # Find unique word counts using dictionary
        training_data = []
        w_target = ""
        for w in tokenizedCorpus:
            for i, word in enumerate(tokenizedCorpus):
                w_target = self.word2onehot(w)
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j>=0 and j<len(tokenizedCorpus):
                        w_context.append(self.word2onehot(tokenizedCorpus[j]))
            training_data.append([w_target, w_context])
        return np.array(training_data)
                
                
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, len(vocabulary))]
        for i, w in enumerate(vocabulary):
            if w == word:
                word_vec[i] = 1
        return word_vec
    
    def train(self, training_data):
        self.w1 = np.random.uniform(-1, 1, (len(vocabulary), self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, len(vocabulary)))
        
        for i in range(self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_pass(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis = 0) # Sum of differences between predicted y and correct w_c in context array corresponding to w_t
                self.backprop(EI, h, w_t)   # Modify w1 and w2 accordingly to the correct w_c's
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c)*np.log(np.sum(np.exp(u)))
            print("Epoch: " + str(i) + " Loss: " + str(self.loss))
            
                
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        self.w1 = self.w1 - self.lr*dl_dw1
        self.w2 = self.w2 - self.lr*dl_dw2
        
    def forward_pass(self, target):
        h = np.dot(self.w1.T, target)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x/e_x.sum(axis = 0)
    
    def vec_sim(self, word, top_n):
        for i in range(len(vocabulary)):
            if vocabulary[i] == word:
                break
        v_w1 = np.array(self.w1[i])
        word_sim = {}
        for m in range(len(vocabulary)):
            if m != i:
                v_w2 = np.array(self.w1[m])
                v_w1a = v_w1.reshape(1, self.n)
                v_w2a = v_w2.reshape(1, self.n)
                cos_lib = cosine_similarity(v_w1a, v_w2a) 
                word = vocabulary[m]
                word_sim[word] = cos_lib
            else:
                continue
        words_sorted = sorted(word_sim.items(), key = lambda kv: kv[1], reverse=True)
        print("Original word: " + word)
        for word1, sim in words_sorted[:top_n]:
            print(word1, sim)
    
    
    
    
    
    
    
    
    
    
#===================================================================================================================
file = open("AC.txt", "r")
corpus = file.read()
tokenized = word_tokenize(corpus)

# Building vocabulary from the corpus
vocabulary = []
weirdlist = [",", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "\""] # The dot "." is not included since it must be used to indicate end of a sentence, which will stop the generator from continuing generating words
for word in tokenized:
    if word not in vocabulary:
        if word not in weirdlist:
            vocabulary.append(word)
        else:
            continue
    else:
        continue
"""
# Build a vocabulary with frequency in corpus
frequency = {}
a = 0
for i in vocabulary:
    for j in tokenized:
        if j == i:
            a += 1
        else:
            continue
    frequency[i] = a
    a = 0
"""
# Defining hyperparameters
settings = {
	'window_size': 2,      	# context window +- center word
	'n': 100,	         	# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 20,	     	# number of training epochs
	'learning_rate': 0.01	# learning rate
}



word2vecAlg = word2vecClass(settings)
trainingData = word2vecAlg.generate_training_data(tokenized)
word2vecAlg.train(trainingData)

word2vecAlg.vec_sim(vocabulary[12], 3)