# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:26:06 2019

@author: Admin
"""

from nltk.tokenize import word_tokenize
import os
import math
import operator

def concatenate(fileDirectory):
    concatenatedCorpus = ""
    entries = os.listdir(fileDirectory)
    for entry in entries:
        file = open(fileDirectory + entry, "r")
        concatenatedCorpus += " " + file.read()
    return concatenatedCorpus

def inputForNaiveBayes(corpus, vocab):
    a = 0
    frequency = {}
    notHelpfulwords = ["a", "the", "this", "that", "those", "these", "his", "her", "of", "to", "for", "--", "is", "it", "be", "and", "or"]
    tokens = word_tokenize(corpus)
    for i in vocab:
        if i in notHelpfulwords:
            continue
        else:
            for j in tokens:
                if j == i:
                    a += 1
                else:
                    continue
        frequency[i] = a
        a = 0
    return frequency


def preprocess(corpus): # Preprocessing concatenated corpora: excluding punctuations, <br /><br />
    weirdlist = [",", "'","\"", "!", "?", ".", "<", "(", ")", "*"]
    official = ""
    j = 0
    while j < (len(corpus)):
        if corpus[j] not in weirdlist and corpus[j:j + 13] != ". <br /><br />" and corpus[j:j + 12] != ".<br /><br />" and corpus[j:j + 11] != "<br /><br />":
            official += corpus[j]
            j += 1
        elif corpus[j:j + 14]== ". <br /><br />":
            j =j + 14
            official += " "
            continue
        elif corpus[j:j + 13] == ".<br /><br />":
            j = j + 13
            official += " "
            continue
        elif corpus[j:j + 12] == "<br /><br />":
            j = j + 12
            official += " "
            continue
        elif corpus[j] in weirdlist:
            j += 1
            continue
        official = official.lower()
    return official

# Building a frequency - vocabulary model for all categories
def frequencyVocab(document):
    vocab = []       
    tokens = word_tokenize(document)
    for word in tokens:
        if word not in vocab:
            vocab.append(word)
        else:
            continue
    freqVocab = inputForNaiveBayes(document, vocab)
    return freqVocab

def trainNaiveBayes(classes): # The argument: classes is an array of directories of available categories: 'neg/' for negative and 'pos/' for positive
    a = {"neg/": 0, "pos/": 0}          # The value are: number of documents of each category
    logPrior = {"neg/": 0, "pos/": 0}   # The value are: logPrior Pc of each category
    overallDoc = ""
    for i in classes: 
        overallDoc += concatenate(i)
    vocabD = frequencyVocab(overallDoc)       # All words in all documents with their frequency
    for i in classes:                       # This loop counts number of doc for each category and store in list a
        entries = os.listdir(i)
        for entry in entries:
            a[i]+=1
    nDocs = a["neg/"] + a["pos/"]
    logLikelihood = {}
    for i in classes:
        bigDoc = concatenate(i)         # All documents of category i
        bigVocab = frequencyVocab(bigDoc)   # Frequency-Vocabuary of all words in all docs of category i
        logPrior[i] = math.log(float(a[i])/nDocs)
        for word in vocabD:
            if word in bigVocab:
                countW = bigVocab[word]     # Number of times word occurs in documents type i
                wDash = 0 + sum(bigVocab.values()) + len(vocabD)    # Summation of all words in the vocabD that are in documents of class i, which is simply 0 plus the number of times words of type i documents occurs
                logLikelihood[word + " of class " + i] = math.log(float(countW + 1)/(wDash + 1))
            else:
                logLikelihood[word + " of class " + i] = math.log(float(1)/(wDash + 1))
                continue         
    return [logPrior, logLikelihood, vocabD]

def testNaiveBayes(testdoc, logPrior, logLikelihood, classes, vocab):
    summation = {}
    for i in classes:
        summation[i] = logPrior[i]
        for word in testdoc:
            if word in vocab:
                summation[i] += summation[i] + logLikelihood[word + " of class " + i]
    return max(summation.items(), key=operator.itemgetter(1))[0]

# Testing here_______________________________________

categories = ["neg/", "pos/"]
overallDoc = ""
for i in categories: 
    overallDoc += concatenate(i)
vocabD = frequencyVocab(overallDoc)
testCorpus = {}
filename = ["testn_79_2.txt", "testn_80_4.txt", "testn_81_3.txt", "testp_1_10.txt", "testp_2_7.txt", "testp_3_7.txt" ]
j = 0;
for i in filename:
    testfile = open(i, "r")
    testCorpus[j] = testfile.read()
    j += 1
for i in range(6):
    a = trainNaiveBayes(categories)
    if testNaiveBayes(testCorpus[i], a[0], a[1], categories, vocabD) == "neg/":
        corpusType = "Negative"
    else:
        corpusType = "Positive"
    print(filename[i] + ": "+ testCorpus[i] + "\n------------> Type: " + corpusType)