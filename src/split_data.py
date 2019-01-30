import re
import json
import numpy as np
from collections import Counter

regex = "[^\w'_]+"

def split_data(data_path):
    with open(data_path) as f:
        data = json.load(f)

    word_count = Counter()
    for dp in data[:10000]:
        text = dp['text'].lower()
        dp['text'] = text.split()
        dp['regex'] = re.split(regex, text)
        dp['is_root'] = 1 if dp['is_root'] else 0
        word_count += Counter(dp['text'])
    for dp in data[10000:]:
        text = dp['text'].lower()
        dp['text'] = text.split()
        dp['regex'] = re.split(regex, text)

    train_data = data[:10000]
    test_data = data[10000:11000]
    validation_data = data[11000:]

    return train_data, test_data, validation_data

def compute_features(data):
    word_count = Counter()
    for dp in data:
        word_count += Counter(dp['text'])
    top_words_160 = word_count.most_common(160)

    for dp in data:
        text = Counter(dp['text'])
        dp['x160'] = [text.get(w, 0) for w, _ in top_words_160]


        "You are my favorite human_being! I'm in love with you_you_you..."
0.06512928009033203