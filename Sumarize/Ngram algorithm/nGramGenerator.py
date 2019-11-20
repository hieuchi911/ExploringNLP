# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:45:32 2019

@author: Admin
"""

from nltk.tokenize import word_tokenize
import random

# 2-gram language model_____________________________
def inputFor2Grams(corpus, vocab):
    a = 0
    frequency = {}
    tokens = word_tokenize(corpus)
    for i in vocab:
        for j in tokens:
            if j == i:
                a += 1
            else:
                continue
        frequency[i] = a
        a = 0
    return frequency
# Calculating probability of 2 words_________________
def probaCalculating(frequency, corpus):
    a = 1
    weirdlist = [",", "'","\"", "!", "?"]
    freq2 = {}
    tokenforuse = []
    tokens = word_tokenize(corpus)
    for word in tokens:                 # This divides the corpus into separate words and stores them in the array: tokenforuse, with all of the punctuations being replaced with an underscore
        if word not in weirdlist:
            tokenforuse.append(word)
        else:
            tokenforuse.append("_")
    for i in range(len(tokenforuse)-2):
        if tokenforuse[i] == "_" or tokenforuse[i+1] == "_":   # If any word in a bigram is a punctualtion (denoted  by "_"), move i to the next word after the punctuation
            continue
        else:
            for j in range(i+1, len(tokenforuse)-1):
                if tokenforuse[j] == tokenforuse[i] and tokenforuse[j+1] == tokenforuse[i+1]:
                    a+=1
                else:
                    j+=1
                    continue
        if tokenforuse[i] + " " + tokenforuse[i+1] not in freq2 :
            freq2[tokenforuse[i] + " " + tokenforuse[i+1]] = [a, (float(a))/(frequency[tokenforuse[i]])]  
        a=1
    if tokenforuse[i+1] + " " + tokenforuse[i+2] not in freq2 and tokenforuse[i+1] != "_" and tokenforuse[i+2] != "_":  # As in the loop with i, i only gets to the length(tokenforuse)-2 index, the last bigram will not be counted in
        freq2[tokenforuse[i+1] + " " + tokenforuse[i+2]] = [1, 1/(frequency[tokenforuse[i+1]])]                         # thus we need this addition if condition
    return freq2

def randomSentenceGenerator(calculatedProba):
    rand = random.random()
    text = ""
    a = 2
    tempBigram = "temporary word"
    for bigram in calculatedProba:
        if abs(calculatedProba[bigram][1] - rand) < 0.222:# and abs(calculatedProba[bigram][1] - rand) > 0.13:
            print(bigram)
            text += bigram
            break
    while a<70:
        rand = random.random()
        for tempBigram in calculatedProba:
            if word_tokenize(tempBigram)[0] == word_tokenize(bigram)[1] and abs(calculatedProba[tempBigram][1] - rand) < 0.27:
                print (word_tokenize(tempBigram)[1])
                text += " " + word_tokenize(tempBigram)[1]
                bigram = tempBigram
        print(text)
        a = a+1
    print(text)
    
# Testing here_______________________________________
    
file = open("AC.txt", "r")
corpus = file.read()
word = ""
vocab = []
weirdlist = [",", "'","\"", "!", "?"] # The dot "." is not included since it must be used to indicate end of a sentence, which will stop the generator from continuing generating words
tokens = word_tokenize(corpus)
for word in tokens:
    if word not in vocab:
        if word not in weirdlist:
            vocab.append(word)
        else:
            continue
    else:
        continue

bigramInput = inputFor2Grams(corpus, vocab)
calculatedProba = probaCalculating(bigramInput, corpus)
randomSentenceGenerator(calculatedProba)