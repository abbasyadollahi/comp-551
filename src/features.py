import re
import json
import numpy as np
from collections import Counter

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

regex = "[^\w'_]+"

def preprocess_data(data_path):

    with open(data_path) as f:
        data = json.load(f)

    for dp in data:
        dp['is_root'] = 1 if dp['is_root'] else 0

    train_set = data[:10000]
    validation_set = data[10000:11000]
    test_set = data[11000:]

    return train_set, validation_set, test_set

#TODO: Implement semantic analysis + other features
#TODO: Dump most common word to words.txt
def compute_features(data):

    word_count = Counter()
    for dp in data:
        text = dp['text']
        dp['num_caps'] = sum(1 for c in text if c.isupper)
        dp['text'] = text.lower().split()
        word_count += Counter(dp['text'])
    top_words_160 = word_count.most_common(160)

    for dp in data:
        dp['no_punct'] = re.split(regex, dp['text'].lower())
        dp['num_words'] = len(dp['no_punct'])
        text = Counter(dp['text'])
        dp['x160'] = [text.get(w, 0) for w, _ in top_words_160]
    
    return data