import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from naive_bayes import NaiveBayes

class LemmaTokenizer:
    '''
    Tokenizes, lemmatizes and removes stop words.
    Pass this as tokenizer for CountVectorizer.
    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        tokens_no_stop = [t for t in tokens if t not in self.stop_words]
        return [self.wnl.lemmatize(t) for t in tokens_no_stop]

def get_vectorizer(binary=False, bigram=False, tfidf=False):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), norm='l2')
    if binary:
        vectorizer.set_params(binary=True)
    if bigram:
        vectorizer.set_params(ngram_range=(1, 2))
    return vectorizer

def naive_bayes_pipeline(bigram=False, tfidf=False):
    vectorizer = get_vectorizer(binary=True, bigram=bigram, tfidf=tfidf)
    nb = NaiveBayes(vectorizer)
    return nb

def log_reg_pipeline(bigram=False, tfidf=False):
    vectorizer = get_vectorizer(bigram=bigram, tfidf=tfidf)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', LogisticRegression(C=1, solver='lbfgs', max_iter=1000, n_jobs=-1))
    ])
    return pipeline

def linear_svc_pipeline(bigram=False, tfidf=False):
    vectorizer = get_vectorizer(bigram=bigram, tfidf=tfidf)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', LinearSVC(C=1, max_iter=10000))
    ])
    return pipeline
