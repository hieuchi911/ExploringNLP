# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 08:31:56 2020

@author: Admin
"""

from word2Vec import word2vecClass
import json
import tensorflow as tf

class inputEmbeddings():
    def __init__(self, context, query): # query here is a list of questions, so we need to traverse through the list
        self.context = context
        self.query = query
        self.contextEmbedding = []
        self.queryEmbedding = []
    
    def generateWord2VecInput(self):
        settings = {
                'window_size': 2,
                'n': 100,
                'epochs': 5,
                'learning_rate': 0.001                
        }
        
        contextW2V = word2vecClass(settings, self.context)
        contextW2V.tokenizeCorpus()
        contextW2V.buildVocabulary()
        contextW2V.train(contextW2V.generate_training_data())
        
        
        queryW2V = word2vecClass(settings, self.query)
        queryW2V.tokenizeCorpus()
        queryW2V.buildVocabulary()
        queryW2V.train(queryW2V.generate_training_data())        
        
        self.contextEmbedding = contextW2V.w1
        self.queryEmbedding = queryW2V.w1
    
if __name__ == "__main__":
    with open("Training and testing data\\light-training-data.json") as f:
        data = json.loads(f.read())
    for topic in data["data"]:
        for context in topic["paragraphs"]:
            for query in context["qas"]:
                print(context["context"])
                print(query["question"])
                inputing = inputEmbeddings(context["context"], query["question"])
                inputing.generateWord2VecInput()
                #contextBiLstmModel = biLstmClass(inputing.contextEmbedding)
                #queryBiLstmModel = biLstmClass(inputing.queryEmbedding)
                tf.keras.layers.Bidirectional()
                
    
    
#embeddings = inputEmbeddings()
#for topic in data["data"]:
#    for context in topic["paragraphs"]:
#        embeddings = inputEmbeddings(context["context"], context["qas"])
    
"""

"""