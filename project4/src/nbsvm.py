import pickle
import numpy as np
import pandas as pd
import os.path as op
from keras import backend as K
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer

from data import load_mr
from pipeline import get_vectorizer

train_data, train_labels, dev_data, dev_labels, test_data, test_labels = load_mr()

print(train_data[:3])
print(train_labels[:3])

def nbsvm_pipeline(max_features, bigram=False, trigram=False, tfidf=False):
    vectorizer = get_vectorizer(max_features, binary=True, bigram=bigram, trigram=trigram, tfidf=tfidf)
    nb = NaiveBayes(vectorizer)
    return nb
