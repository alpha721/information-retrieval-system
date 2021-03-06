from __future__ import unicode_literals
import csv
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
import math
import numpy as np
from collections import Counter
from nltk import *
import time

"""
this code takes in a search query and retrieves the top ten documents relevant to the query

"""


""" builds the lexemes of the corpus by tokenizing, stemming and lemmatizing """
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
      words = []
      words = word_tokenize(doc)
      words = stemmer(words)
      words = lemmatizer(words)
      lexicon.update(word for word in words)
    return lexicon

""" returns the term frequency of the word in the document """
def tf(term, document):
    return freq(term,document)

""" utility function for finding term frequency """
def freq(term,document):
    return document.count(term)

""" function for stemming """
def stemmer(doc):
    stemmed_doc = []
    porter_stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in doc:
        temp = porter_stemmer.stem(word)
        stemmed_doc.append(wordnet_lemmatizer.lemmatize(temp))
    return stemmed_doc

""" function for lemmatizing """
def lemmatizer(doc):
    lemmatized_doc = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in doc:
        lemmatized_doc.append(wordnet_lemmatizer.lemmatize(word))
    return lemmatized_doc

""" calculates the number of document frequncy of a term in the corpus """
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

""" function for finding the inverse document frequency """
def idf(word, doclist):
    n_samples = len(doclist)
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)



""" the original matrix """
doc_term_matrix = []  

""" finds the unit vector corresponding to the vector passed """
def normalizer(vec):
    denom = np.sum([el**2 for el in vec])
    return [(el / math.sqrt(denom)) for el in vec]
 
""" builds the matrix for a given idf_vector """
def build_idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

""" searches a query and returns the cosine score of the documents w.r.t the query """
def search(query_vector,doc_vector):
    
    ratings = [cosine(query_vector, doc) for doc in doc_vector]
    ratings = [i[0] for i in sorted(enumerate(ratings),key=lambda x:x[1], reverse = True)]
    return ratings

""" calculates the cosine similarity of the two vectors """
def cosine(vector1,vector2):
        return float(np.dot(vector1,vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


""" finds the idf-score of the query """
def q_idf(word, query, doclist):
    n_samples = len(doclist)+1
    df = numDocsContaining(word, doclist)
    return np.log(n_samples / 1+df)


    
def main():

    """ Dependency code """
    file = open('Articles.csv')
    file_reader = csv.reader(file)
    file_data = list(file_reader)
    docs = []
    
    #number = len(file_data) 
    number = 10 #testing for 10 documents 
     
    for i in range(0,number):
        docs.append(file_data[i])

    new_docs = []
    for i in range(0,number):
      doc = []
      """ this line is to include all columns of the article into a single document """
      doc = docs[i][0] + " " + docs[i][1] + " " + docs[i][2] + " " + docs[i][3]
      new_docs.append(doc)
     
    docs = new_docs
    
        
    vocab = build_lexicon(docs)
    
    """
    opens the csv file containing tf-idf matrix and reads in into a matrix

    """

    doc_term_matrix_tfidf_l2 = numpy.loadtxt(open("out2.csv","rb"),delimiter=",")
 
    """ read in a query provided by the user """

    query = raw_input("Enter the search query: ")
 
    start = time.time()

    """
    The following functions are used to convert the entered query into a query vector.
    Preprocessing is done by tokenizing, stemming and lemmatizing 
    
    """
    q_lexicon=[] 

    """ Tokenize the query """
    q_lexicon = word_tokenize(query)

    """ Stem the query """
    q_lexicon = stemmer(q_lexicon)
        
    """ Lemmatizes the query """    
    q_lexicon = lemmatizer(q_lexicon)
        
    """ Finds the term frequency in the query """
    q_tf_vector = [tf(word, q_lexicon) for word in vocab]
    
    """ Finds the idf-vector corresponding to the query """
    q_idf_vector = [q_idf(word,q_lexicon,docs) for word in vocab]
    
    q_idf_matrix = build_idf_matrix(q_idf_vector)
    
    """ Finds the tf-idf vector for the query """
    q_tfidf_matrix=normalizer( np.dot(q_tf_vector, q_idf_matrix))

    """ Retrieves the documents in order of their relavency to the query """
    ratings = search(q_tfidf_matrix,doc_term_matrix_tfidf_l2)
   
    """ Displays the top ten documents of the search """
    for i in ratings[0:10]:
        print docs[i] 
    end = time.time()
    
    t = end - start
    
    print "time taken for searching documents: " 
    print t

main()
