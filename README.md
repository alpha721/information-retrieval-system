# information-retrieval-system

This repository contains an implementation of Vector Space Model of Information Retrieval. 
VSM is the backbone of almost all the search engines. This implementation is built on the TF-IDF weighting. 
This information retrieval system targets a corpus containing NEWS articles from 2011.

It uses the dataset "Articles.csv"

Scripts : 
1. corpus_tf_idf.py - finds the tf-idf index for all the documents correspoding to all the terms in the dataset 
2. search.py - computes cosine similarity with a specific query and returns documents ordered according to their relevance. 


Procedure : 

Offline - 
python corpus_tf_idf.py


Online - 
python search.py 

This prompts for the query to be searched, and displays the top ten relevant news articles from the dataset.  



