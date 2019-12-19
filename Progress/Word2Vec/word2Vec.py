# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:24:28 2019

@author: Admin
"""

#from gensim
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#import random

class word2vecClass():
    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']    
    
    def generate_training_data(self, tokenizedCorpus):
        print("In function generate_training_data\n")
        self.wordIndex = {}
        self.indexWord = {}
        m = 0
        for i in vocabulary:
            self.wordIndex[i] = m
            m += 1
        for j in range(len(vocabulary)):
            self.indexWord[j] = vocabulary[j]
        # Find unique word counts using dictionary
        training_data = []
        w_target = []
        print("In for loop of function generate_training_data\n")
        #for w in tokenizedCorpus:
        for sentence in tokenizedCorpus:
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(word)
                w_context = []
                for j in range(i-self.window, i+self.window+1):
                    if j!=i and j>=0 and j<len(sentence):
                        w_context.append(self.word2onehot(sentence[j]))
                        training_data.append([w_target, w_context])
        print("Done for loop of function generate_training_data\n")
        return np.array(training_data)
                
                
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, len(vocabulary))]
        word_vec[self.wordIndex[word]] = 1
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
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            print("Epoch: " + str(i) + " Loss: " + str(self.loss))
        
    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        dl_dw2 = np.outer(h, e)
        # x - shape 1x8, w2 - 5x8, e.T - 8x1
        # x - 1x8, np.dot() - 5x1, dl_dw1 - 8x5
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)
    """
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        randomWords = []
        for i in range(5):
            a = random.randint(0, len(vocabulary)-1)
            randomWords.append(vocabulary[a])
        self.w1[x.index(1)] = self.w1[x.index(1)] - (self.lr*dl_dw1[x.index(1)])
        self.w2[:x.index(1)] = self.w2[:x.index(1)] - (self.lr*dl_dw2[:x.index(1)])
        for word in randomWords:
            self.w1[self.wordIndex[word]] = self.w1[self.wordIndex[word]] - (self.lr*dl_dw1[self.wordIndex[word]])
            self.w2[:self.wordIndex[word]] = self.w2[:self.wordIndex[word]] - (self.lr*dl_dw2[:self.wordIndex[word]])
    """
        
    def forward_pass(self, target):
        h = np.dot(self.w1.T, target)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
        """uSoftmaxed = []
        denominator = 0
        for j in x:
            denominator += np.exp(j)
        for i in x:
            uSoftmaxed.append(np.exp(i)/denominator)
        return uSoftmaxed"""
    
    def vec_sim(self, word, top_n):
        v_w1 = self.w1[self.wordIndex[word]]
        word_sim = {}
        for m in range(len(vocabulary)):
            if vocabulary[m] != word:
                v_w2 = self.w1[self.wordIndex[vocabulary[m]]]
                v_w1a = v_w1.reshape(1, self.n)
                v_w2a = v_w2.reshape(1, self.n)
                cos_lib = cosine_similarity(v_w1a, v_w2a) 
                theWord = vocabulary[m]
                word_sim[theWord] = cos_lib
            else:
                continue
        words_sorted = sorted(word_sim.items(), key = lambda kv: kv[1], reverse=True)
        print("Original word: " + word)
        print("Expected: cats " + str(word_sim["cats"]))
        for word1, sim in words_sorted[:top_n]:
            print(word1, sim)
    
    
    
    
    
    
    
    
    
    
#===================================================================================================================
file = open("testdoc.txt", "r")
corpus = file.read()
print("Finish reading\n")
tokenized = word_tokenize(corpus)
print("Finish tokenizing\n")

# Building vocabulary from the corpus
vocabulary = []
weirdlist = [".", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "\"", "''", "``", ")", "("] # The dot "." is not included since it must be used to indicate end of a sentence, which will stop the generator from continuing generating words
print("Building vocabulary\n\n")
for word in tokenized:
    if word not in vocabulary:
        if word not in weirdlist:
            vocabulary.append(word)
        else:
            continue
    else:
        continue
# Clean tokenized, result in an array of words only
print("Vocabulary has " + str(len(vocabulary)) + " words")
print("Cleaning tokenized corpus.......\n")
weirdlist2 = [",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "''", "``", ")", "("] 
cleanToken = []
temp = []
sentence = []
i = 1
for word in tokenized:
    if word not in weirdlist2:
        temp.append(word)
for word in temp:
    if word != ".":
        sentence.append(word)
    else:
        cleanToken.append(sentence)
        i+=1
        continue
        
print("Finish cleaning\n")
# Defining hyperparameters
settings = {
	'window_size': 2,      	# context window +- center word
	'n': 100,	         	# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 15,	     	# number of training epochs
	'learning_rate': 0.001	# learning rate
}


word2vecAlg = word2vecClass(settings)
print("Generating training data!!\n")
trainingData = word2vecAlg.generate_training_data(cleanToken)
print("Finished generating training data.......\n")
print("Let's train now !!!\n\n")
word2vecAlg.train(trainingData)
print("Done training now wooooohh !!!\n\n")
word2vecAlg.vec_sim("dogs", 3)


#==============================================================================

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