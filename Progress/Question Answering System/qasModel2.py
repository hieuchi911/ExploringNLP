# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:40:27 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:34:14 2020

@author: Admin
path to train file: "/content/drive/My Drive/QaSModel/Question Answering System/Training and testing data/light-training-data.json"

"""

import numpy as np
import tensorflow as tf
import json
from nnWord2vec import word2vec
from nltk.tokenize import word_tokenize
from similarity import Similarity
from context2query import Context2Query
from query2context import Query2Context
from megamerge import MegaMerge
from outputlayer import OutputLayer

class MyQuestionAnsweringModel():
    
    def __init__(self, max_context_length, max_query_length):
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        self.settings = {'window_size': 2, 'n': 100, 'epochs': 5, 'learning_rate': 0.0001}
        
        self.extract_training_inputs("Question Answering System\\Training and testing data")
        
        #input_h = tf.keras.Input(shape = (self.max_context_length, self.settings["n"]), dtype = "float32", name = "context_input")
        #input_u = tf.keras.Input(shape = (self.max_query_length, self.settings["n"]), dtype = "float32", name = "query_input")
        input_h = tf.keras.Input(shape = (114, self.settings["n"]), dtype = "float32", name = "context_input")
        input_u = tf.keras.Input(shape = (20, self.settings["n"]), dtype = "float32", name = "query_input")
        
        # LSTM layer for Context and for Query:
        lstm_context = tf.keras.layers.LSTM(self.settings["n"], recurrent_dropout = 0, return_sequences = True) # Define an LSTM LAYER for context matrix of size dxT
        bidirectional_context_layer = tf.keras.layers.Bidirectional(lstm_context) # Define BiLSTM LAYER by wrapping the lstm_context LAYER
        h = bidirectional_context_layer(input_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        #bidirectional_context_layer_tensor = bidirectional_context_layer(in_h) # context TENSOR of size (1, T, 2d) is returned by plugging an Input tensor context_inputs
        
        
        lstm_query = tf.keras.layers.LSTM(self.settings["n"], recurrent_dropout = 0, return_sequences = True) # Define an LSTM layer for query matrix of size dxT
        bidirectional_query_layer = tf.keras.layers.Bidirectional(lstm_query) # Define BiLSTM LAYER by wrapping the lstm_query LAYER
        u = bidirectional_query_layer(input_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        
        #bidirectional_query_layer_tensor = bidirectional_query_layer(in_u) # query TENSOR of size (1, J, 2d) is returned by plugging an Input tensor query_inputs
        
        #
        #
        #_____________________________ FORMING SIMILARITY MATRIX S _____________________________
        
        # Initiate a 1x6d trainable weight vector with random weights. The shape is 1x6d since this vector will be used in multiplication with concatenated version of outputs from Context (H) and Query (U) biLSTMs: S = alpha(H, U)
        
        similarity_matrix = Similarity()([h, u])
        
        context2query = Context2Query()(u, similarity_matrix)
        
        query2context = Query2Context()(h, similarity_matrix)
        
        #megamerge = MegaMerge()(h, context2query, query2context, self.max_context_length)
        megamerge = MegaMerge()(h, context2query, query2context, 114)
        
        #
        #
        #_____________________________ MODELING LAYER _____________________________
        
        G = tf.transpose(megamerge, [0, 2, 1])  # Transpose G to shape (?, timesteps = num_words, features = 800)
        print("G after expanding dims is: ", G)

        lstm_m1 = tf.keras.layers.LSTM(self.settings["n"], recurrent_dropout = 0, return_sequences = True) # Define an LSTM LAYER for M1
        bidirectional_m1_layer = tf.keras.layers.Bidirectional(lstm_m1) # Define BiLSTM LAYER by wrapping the lstm_m1 LAYER
        m1_tensor = bidirectional_m1_layer(G) # M1 TENSOR of size (?, T, 2d) is returned by plugging an Input tensor G
        
        lstm_m2 = tf.keras.layers.LSTM(self.settings["n"], recurrent_dropout = 0, return_sequences = True) # Define an LSTM LAYER for M1
        bidirectional_m2_layer = tf.keras.layers.Bidirectional(lstm_m2) # Define BiLSTM LAYER by wrapping the lstm_m1 LAYER
        m2_tensor = bidirectional_m2_layer(m1_tensor) # M2 TENSOR of size (?, T, 2d) is returned by plugging an Input tensor m1_tensor 
        
        print("\nm1_tensor is: ", m1_tensor)
        print("m2_tensor is: ", m2_tensor)

        #
        #
        #_____________________________ OUTPUT LAYER _____________________________
        
        # Discharge the first dimension from G, M1 and M2 because they won't be used anymore. Their shape will be (T, 8d), (T, 2d) and (T, 2d) respectively. We next transpose them to
        # coherent shape of (8d, T) and (2d, T)
        G = tf.transpose(G, [0, 2, 1])
        m1_tensor = tf.transpose(m1_tensor, [0, 2, 1])
        m2_tensor = tf.transpose(m2_tensor, [0, 2, 1])

        start_end_index_pred = OutputLayer(name = "output_indices")([G, m1_tensor, m2_tensor])
        

        model = tf.keras.Model(inputs = [input_h, input_u], outputs = start_end_index_pred)
        sgd = tf.keras.optimizers.SGD(learning_rate = 0.001)
        
        model.compile(loss = some_loss_function, optimizer = sgd, metrics = [some_accuracy_metric])
        
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
        for context in data["data"][0]["paragraphs"][0:2]:
            ContextCorpus += " " + context["context"]
            for query in context["qas"]:
                QueryCorpus += " " + query["question"]
                Answer += query["answers"]
        
        print("_______________________CONTEXTS__________________________ \n\n" + ContextCorpus)
        
        print("_______________________QUERIES__________________________ \n\n" + QueryCorpus)
        
        
        self.wordEmbeddingContext = word2vec(self.settings, ContextCorpus)
        self.wordEmbeddingContext.tokenizeCorpus()
        self.wordEmbeddingContext.buildVocabulary()
        self.wordEmbeddingContext.train(self.wordEmbeddingContext.generate_training_data())
        # wordEmbeddingContext.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        self.wordEmbeddingQuery = word2vec(self.settings, QueryCorpus)
        self.wordEmbeddingQuery.tokenizeCorpus()
        self.wordEmbeddingQuery.buildVocabulary()
        self.wordEmbeddingQuery.train(self.wordEmbeddingQuery.generate_training_data())
        # wordEmbeddingQuery.w1 now contains num_word row vectors, each of which has a size of settings["n"]
        
        
        self.count = []
        self.np_Context = []
        self.np_Query = []
        self.Contexts_list = []
        temp_count = 0
        acount = 0
        for context in data["data"][0]["paragraphs"][0:1]:
            # Create word2vec embedding for a Context
            Context = self.tokenizeCorpus(context["context"])  
            self.Contexts_list.append({"tokenized_context": Context, "context_string": context["context"]})
            for word in Context:
                # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                inputContext.append(self.wordEmbeddingContext.w1[self.wordEmbeddingContext.getIndexFromWord(word)].tolist())
            #for q in range(len(Context), self.max_context_length):  # append zeros to the end of each context embedding to gain a coherent size of (max_context_words, n)
            #    inputContext.append(np.zeros(self.settings["n"]))
            inputContext = np.array(inputContext, dtype = np.float32) # inputContext now is a matrix representation of the current context, and it has shape of (num_context_words, n)
            # For each of queries about given Context, create word2vec embedding for that query
            for query in context["qas"]:
                acount += 1
                if acount == 20:
                  break
                temp_count += 1
                Query = self.tokenizeCorpus(query["question"])
                for word in Query:
                    # append a word vector from wordEmbeddingContext by taking the row of w1 at "proper "index. 
                    # Also, w1 shape is (num_words, d = n) so we have to convert np.ndarray to list, so we can subsequently convert the list to tensor
                    inputQuery.append(self.wordEmbeddingQuery.w1[self.wordEmbeddingQuery.wordIndex[word]].tolist()) # shape of inputQuery is (num_words, n)
                for q in range(len(Query), self.max_query_length):# append zeros to the end of each query embedding to gain a coherent size of (max_query_words, n)
                    inputQuery.append(np.zeros(self.settings["n"]))
                inputQuery = np.array(inputQuery, dtype = np.float32) # inputQuery now is a matrix representation of the current query, and it has shape of (num_query_words, n)
                self.np_Query.append({"question": inputQuery, "answer": query["answers"][0]["text"], "query_text": query["question"]})
                self.np_Context.append(inputContext)
                inputQuery = [] # After appending one question matrix, clear it then append the next query matrices
            inputContext = [] # After appending one context matrix, clear it then append the next context matrices
            self.count.append(temp_count)
            print("self.count is: ", self.count)
            temp_count = 0
        print("FINISHED EXTRACTING EMBEDDINGS FOR CONTEXT AND QUERIES, THEY RESPECTIVELY ARE OF SHAPES:\n\n")
        print(np.shape(self.np_Context))
        print(np.shape(self.np_Query[0]["question"]))
        #np_Context = np.asarray(training_context_data, dtype = np.float32) # np_Context here is the context input training_data that will be passed to the fit function of Keras Model
        #self.np_Context = np.transpose(np_Context, (0, 2, 1))
        

    def find_real_answer_indices(self, query_text, query_real_answer, context_index):
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
        
        
        if save1 != 0:
            end_index_real = save1
            print("Answer is: ", self.Contexts_list[context_index]["tokenized_context"][save:save1])
        else:
            end_index_real = save
            print("Answer is: ", self.Contexts_list[context_index]["tokenized_context"][save])
        
        indices_array_1 = np.zeros(self.max_context_length)
        indices_array_1[0] = start_index_real
        
        indices_array_2 = np.zeros(self.max_context_length)
        indices_array_2[0] = end_index_real
        
        print("aray 1 is: ", np.shape(indices_array_1))
        print(str(indices_array_1) + "\n")

        print("aray 2 is: ", np.shape(indices_array_2))
        print(str(indices_array_2) + "\n")
        return [indices_array_1, indices_array_2]
        
    def train(self):
        print("\n========================READY TO TRAIN NOW\n\n\n")
        self.np_Context = np.array(self.np_Context)
        print("np_Context is: ", np.shape(self.np_Context))
        np_Query = []
        for i in self.np_Query:
            print(np.shape(i["question"]))
            np_Query.append(i["question"])
        np_Query = np.array(np_Query)

        print("np_Query is: ", np.shape(self.np_Query))
        
        indices = []
        for i in range(0, len(self.count)):
            for j in range(0, self.count[i]):
                indices_array = self.find_real_answer_indices(self.np_Query[j]["query_text"], self.np_Query[j]["answer"], i) 
                indices.append(indices_array)
        indices = np.array(indices)
        print("indices is: ", np.shape(indices))
        
        self.model.fit({"context_input": self.np_Context, "query_input": np_Query}, {"output_indices": indices}, epochs = self.settings["epochs"], batch_size = 4)
    
    def predict_answer(self, passage, question):
        
        tokenized_passage = self.tokenizeCorpus(passage)
        tokenized_question = self.tokenizeCorpus(question)

        np_Passage = []
        np_Question = []
        i = 0
        index = 0
        for word in tokenized_passage:
            for context in self.Contexts_list:
                if word in context["tokenized_context"]:
                    index += self.count[i]
                    np_Passage.append(self.np_Context[index-1][context["tokenized_context"].index(word)])
                    break
                i += 1
            i = 0
            index = 0
        for word in tokenized_question:
            for question in self.np_Query:
                tokenize = word_tokenize(question["query_text"])
                print("word is: ", word)
                if word in tokenize:
                    print("word is at index:" , tokenize.index(word))
                    np_Question.append(question["question"][tokenize.index(word)])
                    break
        for q in range(len(np_Question), self.max_query_length):# append zeros to the end of each query embedding to gain a coherent size of (max_query_words, n)
            np_Question.append(np.zeros(self.settings["n"]))
        
        predictions = self.model.predict([np.expand_dims(np_Passage, 0), np.expand_dims(np_Question, 0)])
        print("prediction shape is: ", np.shape(predictions))

        start_of_answer = np.argmax(predictions[0][0])
        end_of_answer = np.argmax(predictions[0][1])
        if start_of_answer < end_of_answer:
            print("\nAnswer is: ", self.Contexts_list[i]["tokenized_context"][start_of_answer:end_of_answer])
        else:
            print("\nAnswer is: ", self.Contexts_list[i]["tokenized_context"][start_of_answer])
            print("Or\n\nAnswer is: ", self.Contexts_list[i]["tokenized_context"][end_of_answer])


def some_loss_function(real_answer_indices, prob_start_and_end):
    def extract_loss_start(inputs):
        p1_pred, start_index = inputs
        start_prob = tf.keras.backend.gather(p1_pred, tf.keras.backend.cast(start_index[0], dtype = 'int32'))
        loss = tf.keras.backend.log(start_prob)
        return loss
    def extract_loss_end(inputs):
        p2_pred, end_index = inputs
        end_prob = tf.keras.backend.gather(p2_pred, tf.keras.backend.cast(end_index[0], dtype = 'int32'))
        loss = tf.keras.backend.log(end_prob)
        return loss
    def sum_up_losses(inputs):
        start_loss, end_loss = inputs
        loss1 = start_loss + end_loss
        loss2 = (start_loss + end_loss)*0.5
        loss = tf.where(tf.keras.backend.equal(start_loss, end_loss), loss2, loss1)
        return loss

    p1_pred = prob_start_and_end[:, 0, :]
    
    p2_pred = prob_start_and_end[:, 1, :]
    
    start_index = real_answer_indices[:, 0, :]
    
    end_index = real_answer_indices[:, 1, :]

    start_loss = tf.keras.backend.map_fn(extract_loss_start, (p1_pred, start_index), dtype = tf.float32)
    
    end_loss = tf.keras.backend.map_fn(extract_loss_end, (p2_pred, end_index), dtype = tf.float32)
    
    prob = tf.keras.backend.map_fn(sum_up_losses, (start_loss, end_loss), dtype = tf.float32)    
    
    return -tf.keras.backend.mean(prob, axis = 0)     

def some_accuracy_metric(real_answer_indices, prob_start_and_end):
    def extract_prob_start(inputs):
        p1_pred, start_index = inputs
        start_prob = tf.keras.backend.gather(p1_pred, tf.keras.backend.cast(start_index[0], dtype = 'int32'))
        return start_prob
    def extract_prob_end(inputs):
        p2_pred, end_index = inputs
        end_prob = tf.keras.backend.gather(p2_pred, tf.keras.backend.cast(end_index[0], dtype = 'int32'))
        return end_prob
    def sum_up_probs(inputs):
        start_prob, end_prob = inputs
        loss = (start_prob + end_prob)*0.5
        return loss

    p1_pred = prob_start_and_end[:, 0, :]
    
    p2_pred = prob_start_and_end[:, 1, :]
    
    start_index = real_answer_indices[:, 0, :]
    
    end_index = real_answer_indices[:, 1, :]

    start_pred = tf.keras.backend.map_fn(extract_prob_start, (p1_pred, start_index), dtype = tf.float32)
    
    end_pred = tf.keras.backend.map_fn(extract_prob_end, (p2_pred, end_index), dtype = tf.float32)
    
    acc = tf.keras.backend.map_fn(sum_up_probs, (start_pred, end_pred), dtype = tf.float32)    
    
    return tf.keras.backend.mean(acc, axis = 0)       



if __name__ == "__main__":
    myModel = MyQuestionAnsweringModel(114, 20)
    #myModel.extract_training_inputs("Question Answering System\Training and testing data")
    passage = "Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\"."
    question = "When did Beyonc\u00e9 release Dangerously in Love?"
    myModel.predict_answer(passage, question)
    question2 = "What album made her a worldwide known artist?"
    myModel.predict_answer(passage, question2)
    myModel.train()
    