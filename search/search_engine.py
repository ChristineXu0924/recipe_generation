import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import ast

# get input data, input query, and then return the top 5 most similar recipes

def search(query, corpus):
    vectorizer = TfidfVectorizer().fit(corpus)
    query_vector = vectorizer.transform([query])
    corpus_vector = vectorizer.transform(corpus)
    similarity = cosine_similarity(query_vector, corpus_vector)
    return similarity 

