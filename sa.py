from __future__ import unicode_literals
import csv, os
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
import string
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
    for word in doc:
        stemmed_doc.append(porter_stemmer.stem(word))
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

#the original matrix
doc_term_matrix = []  

def normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
 
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat
    
    
def main():
    file = open('Articles.csv')
    file_reader = csv.reader(file)
    file_data = list(file_reader)
    docs = []
    
    docs.append(file_data[1])
#    docs.append(file_data[2])
#    docs.append(file_data[3])
    #number of documents 
    #number = len(file_data)
    number = 1 #testing for 4 documents 
        
    new_docs = []
    for i in range(0,number):
      doc = []
      # this line is to include all columns of the document as a single thing
      doc = docs[i][0] + " " + docs[i][1] + " " + docs[i][2] + " " + docs[i][3]
      new_docs.append(doc)
     
    docs = new_docs
    
    vocab = build_lexicon(docs)
    for doc in docs:
        lexicon = []            #it contains the tokens( stemmed and lemmatized) from the document under consideration     
        
        #tokenize the words 
        lexicon = word_tokenize(doc)
        
        #stem the words 
        lexicon = stemmer(lexicon)
        
        #lemmatize the words
        lexicon = lemmatizer(lexicon)
        
        tf_vector = [tf(word, lexicon) for word in vocab]
        doc_term_matrix.append(tf_vector)     
    
    #print doc_term_matrix
    
    #normalize the matrix
    normalized_matrix = []
    for vec in doc_term_matrix:
        normalized_matrix.append(normalizer(vec))
    
    #print np.matrix(normalized_matrix)
    
    my_idf_vector = [idf(word, docs) for word in vocab]
    
#    print 'Our vocabulary vector is [' + ', '.join(list(vocab)) + ']'
#    print 'The inverse document frequency vector is [' + ', '.join(format(freq, 'f') for freq in my_idf_vector) + ']'
     
    my_idf_matrix = build_idf_matrix(my_idf_vector)
    
    doc_term_matrix_tfidf = []

    #performing tf-idf matrix multiplication
    for tf_vector in doc_term_matrix:
        doc_term_matrix_tfidf.append(np.dot(tf_vector, my_idf_matrix))

    #normalizing
    doc_term_matrix_tfidf_l2 = []
    for tf_vector in doc_term_matrix_tfidf:
        doc_term_matrix_tfidf_l2.append(normalizer(tf_vector))   
    
#    query = "sindh government public transport"
#    search(query,vocab,docs)
    print vocab
    print np.matrix(doc_term_matrix_tfidf_l2) # np.matrix() just to make it easier to look at

def search(query,vocab,docs):
    lexicon = []
    lexicon = word_tokenize(query)
    lexicon = stemmer(lexicon)
    lexicon = lemmatizer(lexicon)
    query_tf_vector = [tf(word,lexicon) for word in vocab]
    normalized_query_vector = normalizer(query_tf_vector)
    query_idf_vector = [idf(word,docs) for word in lexicon]
    print query_idf_vector

def cosine(vector1,vector2):
    return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))


main()
