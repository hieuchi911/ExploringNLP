# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:40:27 2020

@author: Admin
"""

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
from similarity import Similarity
from context2query import Context2Query
from query2context import Query2Context
from megamerge import MegaMerge
from outputlayer import OutputLayer

class MyQuestionAnsweringModel():
    
    def __init__(self, max_context_length, max_query_length):
        print("IN THE __INIT__ FUNCTION HEYYYYYYYYYY\n\n\n")
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        self.settings = {'window_size': 2, 'n': 100, 'epochs': 5, 'learning_rate': 0.001}
        
        self.extract_training_inputs("Training and testing data\\light-training-data.json")
        
        input_h = tf.keras.Input(shape = (self.max_context_length, self.settings["n"]), dtype = "float32", name = "context_input")
        input_u = tf.keras.Input(shape = (self.max_query_length, self.settings["n"]), dtype = "float32", name = "query_input")
        
        # LSTM layer for Context and for Query:
        lstm_context = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for context matrix of size dxT
        bidirectional_context_layer = tf.keras.layers.Bidirectional(lstm_context) # Define BiLSTM LAYER by wrapping the lstm_context LAYER
        bidirectional_context_layer_tensor = bidirectional_context_layer(input_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        #bidirectional_context_layer_tensor = bidirectional_context_layer(in_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        
        
        lstm_query = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM layer for query matrix of size dxT
        bidirectional_query_layer = tf.keras.layers.Bidirectional(lstm_query) # Define BiLSTM LAYER by wrapping the lstm_query LAYER
        bidirectional_query_layer_tensor = bidirectional_query_layer(input_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        #bidirectional_query_layer_tensor = bidirectional_query_layer(in_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        
        #
        #
        #_____________________________ FORMING SIMILARITY MATRIX S _____________________________
        
        # Initiate a 1x6d trainable weight vector with random weights. The shape is 1x6d since this vector will be used in multiplication with concatenated version of outputs from Context (H) and Query (U) biLSTMs: S = alpha(H, U)
        h = bidirectional_context_layer_tensor # Context TENSOR H with shape (1, num_words, features)
        u = bidirectional_query_layer_tensor # Query TENSOR U with shape (1, num_words, features)
        inputs = {"Context": h, "Query": u}
        similarity_matrix = Similarity()(inputs)
        
        context2query = Context2Query()(u, similarity_matrix)
        
        query2context = Query2Context()(h, similarity_matrix)
        
        megamerge = MegaMerge()(h, context2query, query2context, self.max_context_length)
        #
        #
        #_____________________________ MODELING LAYER _____________________________
        
        G = tf.expand_dims(tf.transpose(megamerge, (1, 0)), 0)  # Transpose G to shape (batch_size, timesteps = num_words, features = 800), then expand one more dimension
                                                        # 0 to plug in the LSTM layer, which corresponds to the batch_size
        lstm_m1 = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for M1
        bidirectional_m1_layer = tf.keras.layers.Bidirectional(lstm_m1) # Define BiLSTM LAYER by wrapping the lstm_m1 LAYER
        m1_tensor = bidirectional_m1_layer(G) # M1 TENSOR of size (1, T, 2d) is returned by plugging an Input tensor G
        
        
        lstm_m2 = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for M1
        bidirectional_m2_layer = tf.keras.layers.Bidirectional(lstm_m2) # Define BiLSTM LAYER by wrapping the lstm_m1 LAYER
        m2_tensor = bidirectional_m2_layer(m1_tensor) # M2 TENSOR of size (1, T, 2d) is returned by plugging an Input tensor m1_tensor
        #
        #
        #_____________________________ OUTPUT LAYER _____________________________
        
        # Discharge the first dimension from G, M1 and M2 because they won't be used anymore. Their shape will be (T, 8d), (T, 2d) and (T, 2d) respectively. We next transpose them to
        # coherent shape of (8d, T) and (2d, T)
        G = tf.transpose(G[0], (1, 0))
        m1_tensor = tf.transpose(m1_tensor[0], (1, 0))
        m2_tensor = tf.transpose(m2_tensor[0], (1, 0))
        G_M1 = tf.transpose(tf.concat((G, m1_tensor), 0), (1, 0)) # G_M1 was of shape (10d, T) then transposed to (T, 10d)
        
        G_M2 = tf.transpose(tf.concat((G, m2_tensor), 0), (1, 0)) # G_M2 was of shape (10d, T) then transposed to (T, 10d)
        
        input_G = {"G_M1": G_M1, "G_M2": G_M2}
        start_end_index_pred = OutputLayer(name = "output_indices")(input_G)
        
        model = tf.keras.Model([input_h, input_u], [start_end_index_pred])
        adam = tf.keras.optimizers.Adam(learning_rate = 0.01)
        
        model.compile(loss = some_loss_function, optimizer = adam, metrics = ["accuracy"])
        
        self.model = model

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
        Answer = []
        
        
        # Now create data to pass as input to the first BiLSTM layer of our model
        # Concatenate all contexts to create word2vec vector:
        a = 0
        for context in data["data"][0]["paragraphs"][0:2]:
            if a==1:
                break
            ContextCorpus += " " + context["context"]
            for query in context["qas"]:
                QueryCorpus += " " + query["question"]
                Answer += query["answers"]
                if a == 3:
                    break
                else:
                    continue
                a+=1
        print("_______________________CONTEXTS__________________________ \n\n" + ContextCorpus)
        
        print("_______________________QUERIES__________________________ \n\n" + QueryCorpus)
        
        self.wordEmbeddingQuery = word2vecClass(self.settings, QueryCorpus)
        self.wordEmbeddingQuery.tokenizeCorpus()
        self.wordEmbeddingQuery.buildVocabulary()
        self.wordEmbeddingQuery.train(self.wordEmbeddingQuery.generate_training_data())
        # wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        
        
        self.wordEmbeddingContext = word2vecClass(self.settings, ContextCorpus)
        self.wordEmbeddingContext.tokenizeCorpus()
        self.wordEmbeddingContext.buildVocabulary()
        self.wordEmbeddingContext.train(self.wordEmbeddingContext.generate_training_data())
        # wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        
        self.count = []
        self.np_Context = []
        self.np_Query = []
        self.Contexts_list = []
        temp_count = 0
        for context in data["data"][0]["paragraphs"][0:1]:
            # Create word2vec embedding for a Context
            Context = self.tokenizeCorpus(context["context"])  
            self.Contexts_list.append({"tokenized_context": Context, "context_string": context["context"]})
            for word in Context:
                # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                inputContext.append(self.wordEmbeddingContext.w1[self.wordEmbeddingContext.getIndexFromWord(word)].tolist())
            for q in range(len(Context), self.max_context_length):  # append zeros to the end of each context embedding to gain a coherent size of (max_context_words, n)
                inputContext.append(np.zeros(self.settings["n"]))
            inputContext = np.asarray(inputContext, dtype = np.float32) # inputContext now is a matrix representation of the current context, and it has shape of (num_context_words, n)
            print("Building context embedding from this context: \n\t" + context["context"] + "\n")
            # For each of queries about given Context, create word2vec embedding for that query
            for query in context["qas"]:
                temp_count += 1
                Query = self.tokenizeCorpus(query["question"])
                for word in Query:
                    # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                    # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                    inputQuery.append(self.wordEmbeddingQuery.w1[self.wordEmbeddingQuery.wordIndex[word]].tolist()) # shape of inputQuery is (num_words, n)
                for q in range(len(Query), self.max_query_length):# append zeros to the end of each query embedding to gain a coherent size of (max_query_words, n)
                    inputQuery.append(np.zeros(self.settings["n"]))
                inputQuery = np.asarray(inputQuery, dtype = np.float32) # inputQuery now is a matrix representation of the current query, and it has shape of (num_query_words, n)
                self.np_Query.append({"question": inputQuery, "answer": query["answers"][0]["text"], "query_text": query["question"]})
                self.np_Context.append(inputContext)
                inputQuery = [] # After appending one question matrix, clear it then append the next query matrices
                print("Building query embedding from this query: \n\t\t" + query["question"])
            inputContext = [] # After appending one context matrix, clear it then append the next context matrices
            self.count.append(temp_count)
            print("self.count is: ", self.count)
            temp_count = 0
        print("FINISHED EXTRACTING EMBEDDINGS FOR CONTEXT AND QUERIES, THEY RESPECTIVELY ARE OF SHAPES:\n\n")
        print(np.shape(self.np_Context))
        print(np.shape(self.np_Query))
        #np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
        #self.np_Context = np.transpose(np_Context, (0, 2, 1))
        

    def find_real_answer_indices(self, query_text, query_real_answer, context_index):
        print("\n\n________________TRAINING WITH THE FOLLOWING CONTEXT-QUERY-ANSWER________________\n\n")
        tokenized = word_tokenize(query_real_answer)
        tokenized_answer = []
        weirdlist = [".", ",", "'","\"", "!", "?", "'", "-", "[", "]", ":", "''", "``", ")", "("]
        for word in tokenized:
            if word not in weirdlist:
                tokenized_answer.append(word)
            else:
                continue
        flag = True
        save = 0
        save1 = 0
        for i in range(0, len(self.Contexts_list[context_index]["tokenized_context"])):
            if self.Contexts_list[context_index]["tokenized_context"][i] == tokenized_answer[0]: # Compare the words in the context at index [context_index] in the self.Contexts_list with the words in the answer tokenized_answer
                save = i
                if len(tokenized_answer) > 1:
                    for j in range(1, len(tokenized_answer)):
                        if self.Contexts_list[context_index]["tokenized_context"][i+j] == tokenized_answer[j]:
                            if j == len(tokenized_answer)-1:
                                save1 = save + len(tokenized_answer)
                                flag = False
                                break
                            else:
                                continue
                        else:
                            break
                    if flag:
                        continue
                    else:
                        break        
                    
        start_index_real = save
        end_index_real = save1
        
        print("\n___Context: ", self.Contexts_list[context_index]["context_string"])
        print("\n___Question: ", query_text)
        print("\n___Correct answer: ", query_real_answer)
        print("\n___Correct answer extracted from indices in the context: ", self.Contexts_list[context_index]["tokenized_context"][save:save1])
        return [start_index_real, end_index_real]
        
    def train(self, indices):
        print("========================READY TO TRAIN NOW\n\n\n")
        np_Context = np.asarray(self.np_Context)
        print("np_Context is: ", np.shape(np_Context))
        np_Query = []
        for i in self.np_Query:
            np_Query.append(i["question"])
        np_Query = np.asarray(np_Query)
        print("np_Query is: ", np.shape(np_Query))
        
        print("indices is: ", np.shape(indices))
        
        
        self.model.fit({"context_input": np_Context, "query_input": np_Query}, {"output_indices": indices})
        
def some_loss_function(real_answer_indices, prob_start_and_end):
    print("\n\n___IN THE LOSS FUNCTION___\nBelow is either called from Compiling phase or from Training phase, if in former phase, these information is abstract, else, it is specific instances\n\n")
    def compute_log_loss(true_and_pred):
        real_answer_indices, p1_pred, p2_pred = true_and_pred
        print("\nTo compare with p1_pred before calling inner function conpute_log_loss, p1_pred is: ", p1_pred)
        print("To compare with p2_pred before calling inner function conpute_log_loss, p2_pred is: ", p2_pred)
        print("\nIn inner function, real_answer_indices is now: ", tf.keras.backend.cast(real_answer_indices[0], dtype = 'int32'))
        real_answer_indices_1 = tf.expand_dims(tf.keras.backend.cast(real_answer_indices[0], dtype = 'int32'), 0)
        real_answer_indices_2 = tf.expand_dims(tf.keras.backend.cast(real_answer_indices[1], dtype = 'int32'), 0)
        print("indices 1:", real_answer_indices_1)
        start_prob = tf.gather(p1_pred, real_answer_indices_1)
        end_prob = tf.gather(p2_pred, real_answer_indices_2)
        
        return -(tf.keras.backend.log(start_prob) + tf.keras.backend.log(end_prob))
    print("\n\nprob_start_and_end is: ", prob_start_and_end)
    print("Before inner function, real start: ", real_answer_indices[0][0])
    print("\t\t\treal end: ", real_answer_indices[0][1])
    p1_pred = tf.expand_dims(prob_start_and_end[0][0], 0) # since prob_start_and_end[0][0] is of size (200,), this when passed to compute_log_loss, will be decreased by 1 dimension, making its shape to be (), and 
    p2_pred = tf.expand_dims(prob_start_and_end[0][1], 0)
    
    print("\nFrom some_loss_function, before calling conpute_log_loss, p1_pred is: ", p1_pred)
    print("From some_loss_function, before calling conpute_log_loss, p2_pred is: ", p2_pred)
    prob = tf.keras.backend.map_fn(compute_log_loss, (real_answer_indices, p1_pred, p2_pred), dtype = 'float32')
    return tf.keras.backend.mean(prob, axis = 0)
    
        


if __name__ == "__main__":
    myModel = MyQuestionAnsweringModel(120, 15)
    indices = []
    for i in range(0, len(myModel.count)):
        for j in range(0, myModel.count[i]):
            indices.append(myModel.find_real_answer_indices(myModel.np_Query[j]["query_text"], myModel.np_Query[j]["answer"], i))
    myModel.train(np.asarray(indices))
    tf.saved_model.save(myModel, "System")