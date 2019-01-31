import os
import re
import json
import numpy as np

from pathlib import Path
from itertools import chain
from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import download as nltk_download


class PreprocessData:

    NUM_TOP_WORDS = 160
    LEXICON = 'vader_lexicon'
    PUNCTUATION_REGEX = "[^\w'_]+"
    DATA_PATH = '../data/proj1_data.json'
    CURSE_WORDS_PATH = '../data/curse_words.txt'
    TOP_WORDS_PATH = '../../words.txt'


    def __init__(self):
        with open(self.DATA_PATH) as f:
            data = json.load(f)

        with open(self.CURSE_WORDS_PATH) as f:
            curses = f.read().splitlines()

        self.data = data
        self.curses = curses
        self.X = []
        self.y = []


    def preprocess_data(self, data):
        for dp in data:
            dp['text_lower'] = dp['text'].lower()
            dp['text_split'] = dp['text_lower'].split()
            dp['text_regex'] = re.split(self.PUNCTUATION_REGEX, dp['text_lower'])
            dp['is_root'] = 1 if dp['is_root'] else 0

        return self.split_data(data)


    def initialize(self, data):
        for dp in data:
            self.X.append([dp['children'], dp['controversiality'], dp['is_root']])
            self.y.append(dp['popularity_score'])


    def split_data(self, data):
        # return data[:10000], data[10000:11000], data[11000:]
        return data[:100], data[100:110], data[110:120]


    def compute_most_common_words(self, data, regex=False):
        word_count = Counter()
        for dp in data:
            word_count += Counter(dp['text_split']) if not regex else Counter(dp['text_regex'])

        top_words = [w[0] for w in word_count.most_common(self.NUM_TOP_WORDS)]
        with open(self.TOP_WORDS_PATH, 'w+') as f:
            f.writelines(f'{word}\n' for word in top_words)


    def feature_most_common_words(self, data):
        with open(self.TOP_WORDS_PATH, 'r+') as f:
            top_words = f.read().splitlines()

        most_common = []
        for dp in data:
            text = Counter(dp['text_lower'])
            most_common.append([text.get(word, 0) for word in top_words])

        return most_common


    def feature_num_curse_words(self, data):
        return [self.count_curses(dp['text_lower']) for dp in data]


    def feature_num_capitals(self, data):
        return [sum(1 for c in dp['text'] if c.isupper()) for dp in data]


    def feature_num_words(self, data):
        return [len(dp['text_split']) for dp in data]


    def feature_sentiment(self, data):
        return [self.sentiment_analysis(dp['text'])['compound'] for dp in data]


    def feature_links(self, data):
        url = ['http', 'www', '.com', '.ca']
        return [1 if any(s in dp['text_lower'] for s in url) else 0 for dp in data]


    def compute_features(self, data):
        self.initialize(data)

        top_words = self.feature_most_common_words(data)
        num_curse_words = self.feature_num_curse_words(data)
        num_capitals = self.feature_num_capitals(data)
        num_words = self.feature_num_words(data)
        sentiment = self.feature_sentiment(data)
        links = self.feature_links(data)

        for i, x in enumerate(self.X):
            x += top_words[i]
            x.append(num_curse_words[i])
            x.append(num_capitals[i])
            x.append(num_words[i])
            x.append(sentiment[i])
            x.append(links[i])

        X = np.array(self.X)
        y = np.array(self.y)

        return X, y


    def count_curses(self, words):
        return sum(1 for curse in self.curses if curse in words)


    def sentiment_analysis(self, text):
        if not os.path.exists(f'./sentiment/{self.LEXICON}.zip'):
            nltk_download('vader_lexicon', download_dir='.')

        sia = SentimentIntensityAnalyzer()
        ps = sia.polarity_scores(text)
        return ps


ppd = PreprocessData()
train, validation, test = ppd.preprocess_data(ppd.data)
ppd.compute_most_common_words(train)
X, y = ppd.compute_features(train)
for i in range(5):
    print(X[i])
    print(y[i])
    print(train[i])
