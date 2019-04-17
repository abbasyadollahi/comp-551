import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nbsvm import build_nbsvm

class LemmaTokenizer:
    '''
    Tokenizes, lemmatizes and removes stop words.
    Pass this as tokenizer for CountVectorizer.
    '''

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return tokens
        # return [self.wnl.lemmatize(t) for t in tokens]

def get_vectorizer(max_features=None, binary=False, ngram=1, tfidf=False):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, binary=binary, ngram_range=(1, ngram))
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, ngram_range=(1, ngram), norm='l2')
    return vectorizer

def linear_svc_pipeline(max_features=None, ngram=1, tfidf=False):
    vectorizer = get_vectorizer(max_features, ngram=ngram, tfidf=tfidf)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', LinearSVC(loss='squared_hinge', C=1, tol=1e-4, max_iter=1000))
    ])
    return pipeline

def nbsvm_pipeline(data, labels, max_features=None, ngram=1, tfidf=False):
    vectorizer = get_vectorizer(max_features, binary=True, ngram=ngram, tfidf=tfidf)
    dtm_train = vectorizer.fit_transform(data)
    num_words = len(vectorizer.vocabulary_) + 1

    pipeline, x_train = build_nbsvm(dtm_train, labels, num_words)
    return pipeline, x_train
