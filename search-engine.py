import csv, os
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
    return document.split().count(term)

vocabulary = set()

doc_term_matrix = []

print 'Our Vocabulary vector is [' + ', '.join(list(vocabulary)) + ']'

def main():
    file = open('Articles.csv')
    file_reader = csv.reader(file)
    file_data = list(file_reader)
#    print file_data[0]
    docs = []
    docs.append(file_data[1])
    docs.append(file_data[2])
#    print docs[0]
#    word = word_tokenize(docs[0])
#    print word
    docs[0][0] += " by by by by"
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
        doc = docs[doc_id][0] + " " + docs[doc_id][1] + " " +  docs[doc_id][2] + " " +  docs[doc_id][3]
        #doc = docs[doc_id]
        print 'doc is : ', doc
        lexicon.update(word_tokenize(doc))
   #    print lexicon
        for word in lexicon:
           doc_ref = {}
           doc_ref[doc_id] = tf(word,doc)
           if word not in table:
               table[word] = []
           table[word].append(doc_ref)
    for key in table:
        print key, ": " , table[key]



main()
