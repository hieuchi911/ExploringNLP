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
import math
from similarity import Similarity
from context2query import Context2Query
from query2context import Query2Context
from megamerge import MegaMerge
from outputlayer import OutputLayer

class MyQuestionAnsweringModel():
    
    def __init__(self):
        super(MyQuestionAnsweringModel, self).__init__()
        self.settings = {'window_size': 2, 'n': 100, 'epochs': 5, 'learning_rate': 0.001}
        
        trainable_weight_vector = np.random.random([1, 6*self.settings["n"]]).astype(np.float32)
        self.w1 = tf.Variable(trainable_weight_vector) # trainable weight vector w1
        
        
        trainable_weight_vector_p1 = np.random.random([1, 10*self.settings["n"]]).astype(np.float32)
        self.wp1 = tf.Variable(trainable_weight_vector_p1) # trainable weight vector wp1
        
        trainable_weight_vector_p2 = np.random.random([1, 10*self.settings["n"]]).astype(np.float32)
        self.wp2 = tf.Variable(trainable_weight_vector_p2) # trainable weight vector wp2
        

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
                Answer += query["answers"]
                if a == 3:
                    break
                else:
                    continue
                a+=1
        print("_______________________CONTEXTS__________________________ \n\n" + ContextCorpus)
        print(type(ContextCorpus))
        
        print("_______________________QUERIES__________________________ \n\n" + QueryCorpus)
        print(type(QueryCorpus))
        
        self.wordEmbeddingContext = word2vecClass(self.settings, ContextCorpus)
        self.wordEmbeddingContext.tokenizeCorpus()
        self.wordEmbeddingContext.buildVocabulary()
        self.wordEmbeddingContext.train(self.wordEmbeddingContext.generate_training_data())
        # wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        
        self.wordEmbeddingQuery = word2vecClass(self.settings, QueryCorpus)
        self.wordEmbeddingQuery.tokenizeCorpus()
        self.wordEmbeddingQuery.buildVocabulary()
        self.wordEmbeddingQuery.train(self.wordEmbeddingQuery.generate_training_data())
        # wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        self.np_Context = []
        self.np_Query = []
        self.Contexts_list = []
        for i in range(1):
            for context in data["data"][0]["paragraphs"][0:1]:
                
                # Create word2vec embedding for a Context
                Context = self.tokenizeCorpus(context["context"])  
                self.Contexts_list.append({"tokenized_context": Context, "context_string": context["context"]})
                for word in Context:
                    # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                    # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                    inputContext.append(self.wordEmbeddingContext.w1[self.wordEmbeddingContext.getIndexFromWord(word)].tolist())
                inputContext = np.asarray(inputContext, dtype = np.float32) # inputContext now is a matrix representation of the current context, and it has shape of (num_context_words, n)
                self.np_Context.append(inputContext)
                inputContext = [] # After appending one context matrix, clear it then append the next context matrices
                print(context["context"])
                # For each of queries about given Context, create word2vec embedding for that query
                for query in context["qas"]:
                    Query = self.tokenizeCorpus(query["question"])
                    for word in Query:
                        # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                        # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                        inputQuery.append(self.wordEmbeddingQuery.w1[self.wordEmbeddingQuery.wordIndex[word]].tolist()) # shape of inputQuery is (num_words, n)
                    inputQuery = np.asarray(inputQuery, dtype = np.float32) # inputQuery now is a matrix representation of the current query, and it has shape of (num_query_words, n)
                    self.np_Query.append({"question": inputQuery, "answer": query["answers"][0]["text"], "start index": query["answers"][0]["answer_start"]})
                    
                    inputQuery = [] # After appending one question matrix, clear it then append the next query matrices
                self.np_Query.append(np.array([0]))     # This indicates the end of the questions series related to the context 
                 
            #np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
            #self.np_Context = np.transpose(np_Context, (0, 2, 1))
    
#def train(context_train, query_train, query_real_answer, context_index, answer_start_char_index): # Each of the argument here has a shape of (num_words, n), num_words will be equal to the number of time steps, n will be equal to num_features in the input of LSTM
    
    def call(self, context_train, query_train):        
        #
        #
        #_____________________________ DEFINE INPUTS EMBEDDINGS USING BiLSTMs _____________________________
        print(type(context_train))
        # Context and query inputs to LSTM layers must each be a 3D tensor: 3D are 1 being batch size, num_words being the number of timesteps, 200 being number of features
        #in_h = tf.convert_to_tensor(context_train, name = "context_inputs")
        in_h = tf.expand_dims(context_train, 0)  # Expand one more dimension to plug in the LSTM layer, which corresponds to the batch_size
        print(in_h.get_shape())
        #in_u = tf.convert_to_tensor(query_train, name = "query_inputs")
        in_u = tf.expand_dims(query_train, 0)  # Expand one more dimension to plug in the LSTM layer, which corresponds to the batch_size
        print(in_u.get_shape())
        
        
        # LSTM layer for Context and for Query:
        lstm_context = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for context matrix of size dxT
        bidirectional_context_layer = tf.keras.layers.Bidirectional(lstm_context) # Define BiLSTM LAYER by wrapping the lstm_context LAYER
        #bidirectional_context_layer_tensor = bidirectional_context_layer(input_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        bidirectional_context_layer_tensor = bidirectional_context_layer(in_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        
        
        lstm_query = tf.keras.layers.LSTM(self.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM layer for query matrix of size dxT
        bidirectional_query_layer = tf.keras.layers.Bidirectional(lstm_query) # Define BiLSTM LAYER by wrapping the lstm_query LAYER
        #bidirectional_query_layer_tensor = bidirectional_query_layer(input_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        bidirectional_query_layer_tensor = bidirectional_query_layer(in_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        
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
        
        megamerge = MegaMerge()(h, context2query, query2context)
        
                            
        """   # S will be the similarity matrix, is expected to have shape (num_context_words, num_query_words)
                    i_to_j_relateness = []  # This is a temporary array that stores a vector of scalars denoting similarity between all query words and one word i
                    m = 0
                    count = 0 # This variable is used to count the number of Context words
                    for i in h[0]: # Index of i is the corresponding index of the word in the context
                        count += 1
                        for j in u[0]: # Index of j is the corresponding index of the query word
                            
                            # i and j are of size (200,), transposing them to make them row vectors, then concatenate them with the element-wise multiplication of themselves to earn temp
                            temp = tf.concat((i, j, tf.math.multiply(i, j)), 0) # temp shape is (600,) so we have to expand it to (600, 1) -> Use expand_dims
                            temp = tf.expand_dims(temp, 1)
                            # self.w1 is of shape (1, 6d) and temp is of shape (6d, 1) -> alpha is of shape (1, 1)
                            alpha = tf.tensordot(self.w1, temp, 1) # The dot product returns a scalar representing similarity between the "words" i and j ( i and j arent words, but they decodes words)
                            if m == 0:
                                i_to_j_relateness = alpha
                            else:
                                i_to_j_relateness = tf.concat((i_to_j_relateness, alpha), 1) # add the scalars alpha in, the loop ends and results in i_to_j_relateness being the similarity matrix between the word i and all the words in the query
                            m +=1
                        if count == 1:
                            S = i_to_j_relateness
                        else:
                            S = tf.concat((S, i_to_j_relateness), 0)
                        i_to_j_relateness = 0
                        m = 0
                    
                    #S = np.array(S) # Turn S from a list to an ndarray of size (num_context_words, num_query_words)
                    #
                    #
                    #_____________________________ FORMING CONTEXT TO QUERY MATRIX FROM S _____________________________
                    
                    A = tf.keras.activations.softmax(S, axis = 1) # A, of size (num_context_words, num_query_words), is the distribution of similarities between of words in context and in queries
                    m = 0
                    count1 = 0
                    for i in A: # i is of shape (1, num_query_words)
                        for j in i: # j is a scalar
                            if m == 0:
                                sum_of_weighted_query = tf.math.scalar_mul(j, u[0][m]) # sum_of_weighted_query is of shape (2d = 200)
                            else:
                                sum_of_weighted_query += tf.math.scalar_mul(j, u[0][m])
                            m +=1
                        m = 0
                        count1 += 1
                        if count1 == 1:
                            c2q = tf.expand_dims(sum_of_weighted_query, 0)
                        else:
                            c2q = tf.concat((c2q, tf.expand_dims(sum_of_weighted_query, 0)), 0) # c2q is expected to be of shape (200, num_context_words) -> need to take transpose
                    c2q = tf.transpose(c2q) # c2q is now a tensor of size (200, T), encapsulates the RELEVANCE of each Query word to each Context word
                    #
                    #
                    #_____________________________ FORMING QUERY TO CONTEXT MATRIX FROM S _____________________________
                    
                            # z, of size (1, num_context_words) is a vector whose elements are each max of the corresponding row in similarity matrix S. z encapsulates the Query word that is most
                            # relevant to each  Context word (here, only Context words that are really relevant to a query will be shown in z by their high similarity value, and for 
                            # Context words that cannot pay tribute to the answer, they will be neglect and assigned values close to zero)
                    
                    z1 = tf.keras.backend.max(S, axis = -1) # Take max at each row to generate a column vector z1
                    b = tf.nn.softmax(z1) # apply softmax on all elements of z and store in b. b is of shape (1, num_context_words)
                    m = 0
                    q2c = []
                    for scalar in b:
                        if m == 0:
                            sum_of_weighted_context = tf.math.scalar_mul(scalar, h[0][m])   # Scalar is achieved from b (or z), if it is low, then when being multiplied with 
                                                                                            # each (1, 200) vector of h (this vector corresponds to the contextual representation
                                                                                            # of that Context word), the scalar causes a decrease in that word's vector, hence decrease
                                                                                            # its contribution to the sum_of_weighted_context. But if it is high, then the corresponding vector
                                                                                            # being multiplied with will pay much contribution to sum_of_weighted_context. Then, the accumulated vector
                                                                                            # sum_of_weighted_context now represents important Context words that answer the Query. Being duplicated
                                                                                            # to form q2c, q2c will now decodes the information about most important Context words.
                        else:
                            sum_of_weighted_context += tf.math.scalar_mul(scalar, h[0][m])
                        m += 1
                    m = 0
                    sum_of_weighted_context = tf.expand_dims(sum_of_weighted_context, 1)
                    for scalar in b:
                        if m == 0:
                            q2c = sum_of_weighted_context
                        else:
                            q2c = tf.concat((q2c, sum_of_weighted_context), 1) # duplicate the row vector sum_of_weighted_context of shape (200,) for num_words_query times
                        m += 1
                      # q2c is of size (200, T), encapsulates information about the most important words in the Context w.r.t the Query
                    #
                    #
                    #_____________________________ MEGAMERGING MATRICES: H, C2Q AND Q2C _____________________________
                    
                    # At this point, H is of shape(1, num_context_words, 200); C2Q and Q2C are of shape(200, num_context_words)
                    for c in range(0, count): # count is the number of context words in Context
                        # beta below is a row vector of size (1, 800)
                        beta = tf.concat((tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(c2q)[c], 1), tf.math.multiply(tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(c2q)[c], 1)), tf.math.multiply(tf.expand_dims(h[0][c], 1), tf.expand_dims(tf.transpose(q2c)[c], 1))), 0) # 0 here indicates that concatenation is done along rows, since each element in the concatenate func is of size (200,)
                        if c == 0:
                            G = beta                    # G, after being transposed to be of size (800, num_context_words), is the QUERY-AWARE REPRESENTATION of ALL CONTEXT WORDS, where each column
                                                        # vector is a concatenation of the Context word itself, the query-context-relevance subvector and the importance-to-query
                                                        # subvector. Each column of G is the representation of a Context word that is aware of the existence of the Query and
                                                        # has incorporated-and-relevant information from the Query.
                        else:
                            G = tf.concat((G, beta), 1)"""
        #
        #
        #_____________________________ MODELING LAYER _____________________________
        
        G = tf.expand_dims(tf.transpose(megamerge, (1, 0)), 0)  # Transpose G to shape (batch_size, timesteps = num_words, features = 800), then expand one more dimension
                                                        # 0 to plug in the LSTM layer, which corresponds to the batch_size
        lstm_m1 = tf.keras.layers.LSTM(myModel.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for M1
        bidirectional_m1_layer = tf.keras.layers.Bidirectional(lstm_m1) # Define BiLSTM LAYER by wrapping the lstm_m1 LAYER
        m1_tensor = bidirectional_m1_layer(G) # M1 TENSOR of size (1, T, 2d) is returned by plugging an Input tensor G
        
        
        lstm_m2 = tf.keras.layers.LSTM(myModel.settings["n"], activation = "tanh", trainable = False, return_sequences = True) # Define an LSTM LAYER for M1
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
        start_index_pred, end_index_pred = OutputLayer()(input_G)
        """m = 0
                            for i in G_M1:
                                scalar = tf.tensordot(self.wp1, tf.expand_dims(i, 1), 1)
                                if m == 0:
                                    # i is of shape (10d, )
                                    p1 = scalar
                                else:
                                    p1 = tf.concat((p1, scalar), 1)
                                m += 1
                            m = 0
                            for i in G_M2:
                                scalar = tf.tensordot(self.wp2, tf.expand_dims(i, 1), 1)
                                if m == 0:
                                    p2 = scalar
                                else:
                                    p2 = tf.concat((p2, scalar), 1)
                                m += 1
                            print("p1 is: ", p1)
                            print("p2 is: ", p2)
                            p1_pred = tf.nn.softmax(p1)
                            p2_pred = tf.nn.softmax(p2)
                            
                            print("p1_pred is: ", p1_pred)
                            print("p2_pred is: ", p2_pred)"""
        
        return start_index_pred, end_index_pred

def train(model, input_context, input_queries, query_real_answer, context_index):
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
    for i in range(0, len(model.Contexts_list[context_index]["tokenized_context"])):
        if model.Contexts_list[context_index]["tokenized_context"][i] == tokenized_answer[0]:
            save = i
            if len(tokenized_answer) > 1:
                for j in range(1, len(tokenized_answer)):
                    if model.Contexts_list[context_index]["tokenized_context"][i+j] == tokenized_answer[j]:
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
    
    print("Real answer is: ", model.Contexts_list[context_index]["tokenized_context"][save:save1])
    
    with tf.GradientTape() as tape:
        model_loss = some_loss_function(model(input_context, input_queries), start_index_real, end_index_real)
        #model_loss += sum(model.losses)
        
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3)
    model_gradients = tape.gradient(tf.convert_to_tensor(model_loss), model.trainable_variables)
    optimizer.apply_gradients(zip(model_gradients, myModel.trainable_variables))
        
def some_loss_function(prob_start_and_end, real_answer_start_index, real_answer_end_index):
    print("real start: ", real_answer_start_index)
    print("real end: ", real_answer_end_index)
    return -(math.log(float(prob_start_and_end[0][0][real_answer_start_index])) + math.log(float(prob_start_and_end[1][0][real_answer_end_index])))/50
    
        
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

myModel = MyQuestionAnsweringModel()
    
if __name__ == "__main__":
    myModel.extract_training_inputs("Training and testing data\\light-training-data.json")
    context_index = 0
    for i in myModel.np_Context:
        for j in myModel.np_Query:
            if np.shape(j) == (1,): # This is when j is an array of size (1,), which indicates that the next j's are for the next context, so we must change context (change i)
                print("break!!!")
                break
            else:
                train(myModel, i, j["question"], j["answer"], context_index)
                break
        context_index += 1
                
    
"""
x = tf.keras.layers.concatenate([bidirectional_context_layer_tensor, bidirectional_query_layer_tensor])

model = tf.keras.Model(inputs = [context_inputs, query_inputs], outputs = x)
model.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3), loss = tf.keras.losses.BinaryCrossentropy(from_logits=True))
model.fit({"context_inputs": np.transpose(np_Context, (0, 2, 1)), "query_inputs": np.transpose(np_Query[1], (0, 2, 1))}, epochs = 3)
#S = 
"""