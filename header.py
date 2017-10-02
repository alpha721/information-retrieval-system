from __future__ import unicode_literals
import csv
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
import math
import numpy as np
from collections import Counter
from nltk import *


def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
      words = []
      words = word_tokenize(doc)
      words = stemmer(words)
      words = lemmatizer(words)
      lexicon.update(word for word in words)
#        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
    return freq(term,document)

def freq(term,document):
    return document.count(term)


def stemmer(doc):
    stemmed_doc = []
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in doc:
        temp = porter_stemmer.stem(word)
        stemmed_doc.append(wordnet_lemmatizer.lemmatize(temp))
    return stemmed_doc


def lemmatizer(doc):
    lemmatized_doc = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in doc:
        lemmatized_doc.append(wordnet_lemmatizer.lemmatize(word))
    return lemmatized_doc

def numDocsContaining(word, doclist):
    doccount = 0
    for doc in doclist:
        lexicon = []  
        lexicon = word_tokenize(doc)
        lexicon = stemmer(lexicon)
        lexicon = lemmatizer(lexicon)
        if ( freq(word, lexicon) > 0):
          doccount += 1
    return doccount 

def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)

def q_idf(word, query, doclist):
    n_samples = len(doclist)+1
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)


#the original matrix
doc_term_matrix = []  

def normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
 
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

def search(query_vector,doc_vector):
    
    ratings = [cosine(query_vector, doc) for doc in doc_vector]
    ratings = [i[0] for i in sorted(enumerate(ratings),key=lambda x:x[1], reverse = True)]

    
    #ratings.sort(reverse = True)
    return ratings

def cosine(vector1,vector2):
        return float(np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))



