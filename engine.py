from __future__ import unicode_literals
import csv, os
import sys
reload(sys)
sys.setdefaultencoding('ISO-8859-1')
import string
from collections import Counter
from nltk import *

table = {}


#def indexing(docs):
#    
#    for doc in docs:
#        lexicon = set()
#        lexicon.update([word_tokenize(doc)])
#        processed_words = set()
#        for word in lexicon:
#            word = porter_stemmer.stem(word)
#            word = wordnet_lemmatizer.lematize(word)
#            processed_words.update(word if word not in stop_words)
#            hashed_docs = table(word)
#            if doc not in hashed_docs:
#                table(word).append(doc)
#
#

def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])

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

#vocabulary = set()

#doc_term_matrix = []

#print 'Our Vocabulary vector is [' + ', '.join(list(vocabulary)) + ']'

