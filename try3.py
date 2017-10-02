# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:04:52 2017

@author: sys pc
"""

from __future__ import division, unicode_literals
import math
from textblob import TextBlob as tb

import pandas as pd 
from collections import Counter
import csv

def corpus_counter(corpus):
    holder = []
    counter = 0
    for i in corpus:
        x = i.split()

        # get a set of each question
        # this will be used to count the number of documents contains a specific word
        x = list(set(x))

        holder += x
        counter += 1
    print "Number of documents accounted for =", counter
    return holder

# get a bloblist for the text using textblob.
def make_bloblist(column):
    holder = []
    for i in column:
        doc = tb(i)
        holder.append(doc)
    return holder



df = pd.read_csv(('Articles.csv'))

#Some column info for reference

df1=df[['Article','Heading','NewsType'] ][3:100]
df1.info()

doc_counter = corpus_counter(df1)


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1+(n_containing(word, bloblist))))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


bloblist = make_bloblist(df1.Heading)

for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:1]:
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
        
        
        
with open("C:\\Users\sys pc\Desktop\\titles.csv", "w") as file:
    writer = csv.writer(file, delimiter=str(u','))
    writer.writerow(["Id", "Title", "Score"])
 
    doc_id = 0
    for doc in enumerate(bloblist):
        print "Document %d" %(doc_id)
        word_id = 0
        for scores in sorted_words[:]:
            if scores > 0:
                title = sorted_words[word_id][0]
                writer.writerow([doc_id+1, title, scores[1]])
            word_id +=1
        doc_id +=1    
