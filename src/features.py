import os
import re
import json
import numpy as np

from collections import Counter
from pathlib import Path

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download

punct_regex = "[^\w'_]+"
num_most_common = 160
lexicon = 'vader_lexicon'
swear_path = './data/curse_words.txt'
data_path = './data/proj1_data.json'

def preprocess_data(path):

    with open(path) as f:
        data = json.load(f)

    for dp in data:
        dp['is_root'] = 1 if dp['is_root'] else 0

    # train_set = data[:10000]
    # validation_set = data[10000:11000]
    # test_set = data[11000:]

    # DEBUG
    train_set = data[:100]
    validation_set = data[100:110]
    test_set = data[110:120]

    return train_set, validation_set, test_set

#TODO: Dump most common word to words.txt
def compute_features(data):
    swears = load_swears(swear_path)

    word_count = Counter()
    word_count_regex = Counter()
    for dp in data:
        text = dp['text']

        sentiment = sentiment_analysis(text)
        dp['sentiment'] = sentiment['compound']

        dp['num_caps'] = sum(1 for c in text if c.isupper)

        # Should we remove empty entries when it splits at end of string
        text = text.lower()
        dp['no_punct'] = re.split(punct_regex, text)

        dp['text'] = text.split()
        word_count += Counter(dp['text'])
        word_count_regex += Counter(dp['no_punct'])

    top_words = [w[0] for w in word_count.most_common(num_most_common)]

    for dp in data:
        dp['num_words'] = len(dp['no_punct'])

        dp['num_swears'] = count_swears(dp['no_punct'], swears)

        text = Counter(dp['text'])
        dp['most_common'] = [text.get(w, 0) for w in top_words]

    features = []
    for dp in data:
        features.append([dp['children'], dp['controversiality'], dp['is_root'], dp['num_words'], dp['sentiment'], dp['num_caps'], dp['num_swears']] + dp['most_common'])
    X = np.array(features)
    y = np.array([dp['popularity_score'] for dp in data])

    return X, y

def check_lexicon():
    if not os.path.exists(os.path.join(Path.home(), f'nltk_data/sentiment/{lexicon}.zip')):
        nltk_download('vader_lexicon')

def sentiment_analysis(text):
    check_lexicon()

    sia = SentimentIntensityAnalyzer()
    ss = sia.polarity_scores(text)
    return ss

def load_swears(path):
    with open(swear_path) as f:
        swears = f.readlines()
    return swears

def count_swears(words, swears):
    num_swears = sum(1 for w in words if w in swears)
    return num_swears

train, validation, test = preprocess_data(data_path)
X, y = compute_features(train)

print(y)
print(y.shape)
print(X.shape)