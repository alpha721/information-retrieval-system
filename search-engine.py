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

def main():
    file = open('Articles.csv')
    file_reader = csv.reader(file)
    file_data = list(file_reader)
#    print file_data[0]
    docs = []
    docs.append(file_data[1])
    docs.append(file_data[2])
    docs.append(file_data[3])
#    print docs[0]
#    word = word_tokenize(docs[0])
#    print word
    docs[0][0] += " by by by by"
    docs[1][0] += " by by by by"
#    print docs[0]
#    vocabulary = build_lexicon(docs)

#    for doc in docs:
#        print 'The doc is " ' + doc + ' " '
#        tf_vector = [tf(word,doc) for word in vocabulary]
#        tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
#        print 'the tf vector for Document %d is [%s]' % ((docs.index(doc) + 1), tf_vector_string)
#        doc_term_matrix.append(tf_vector)
#
#    print 'master document term matrix: '
#    print doc_term_matrix

    n = len(docs)
    print 'size of docs : ' , n

    
#    for doc_id in range(0,n):
#        doc = docs[doc_id]
#        lexicon = set()
#        print 'doc is : ' , doc
#        lexicon.update(word_tokenize(doc))
#        print lexicon
#        print doc.split()
#        print 'count of "by" :'
#        print doc.split().count("by")
#        for word in lexicon:
#            doc_ref = {}
#            doc_ref[doc_id] = tf(word,doc)
#            if word not in table:
#                table[word] = []
#            table[word].append(doc_ref)

#    for key in table:
#        print key, ": " , table[key]


    for doc_id in range(0,n):
        lexicon = set()
        bag_of_words = []
        doc = docs[doc_id][0] + " " + docs[doc_id][1] + " " +  docs[doc_id][2] + " " +  docs[doc_id][3]
        #doc = docs[doc_id]
   #     print 'doc is : ', doc
        lexicon.update(word_tokenize(doc))
        lexicon = stemmer(lexicon)
        lexicon = lemmatizer(lexicon)
        bag_of_words = word_tokenize(doc)
        bag_of_words = stemmer(bag_of_words)
        bag_of_words = lemmatizer(bag_of_words)

   #    print lexicon
        for word in lexicon:
           doc_ref = {}
           doc_ref[doc_id] = tf(word,bag_of_words)
           if word not in table:
               table[word] = []
           table[word].append(doc_ref)
    for key in table:
        print key, ": " , table[key]


    query = "karachi public transport in by"
    query = query.lower()
    query = word_tokenize(query)
    query = stemmer(query)
    query = lemmatizer(query)
#    new_query = []
#    for token in query:
#        if token not in stop_words:
#            new_query.append(token)
#    query = new_query
#
#    for token in query:
#        for i in range(0,len(table[token])):
#            doc_ids = map(token,[table[token][i].keys()])
#    print doc_ids

#    for i in range(0,len(table['by'])):
#        print table['by'][i].keys()

main()
