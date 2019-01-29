import re
import json
from collections import Counter

with open('proj1_data.json') as f:
    data = json.load(f)

word_count = Counter()
for dp in data[:10000]:
    dp['text'] = dp['text'].lower().split()
    word_count += Counter(dp['text'])
for dp in data[10000:]:
    dp['text'] = dp['text'].lower().split()

train_data = data[:10000]
test_data = data[10000:11000]
validation_data = data[11000:]

top_words_160 = word_count.most_common(160)
for dp in data:
    text = Counter(dp['text'])
    dp['x160'] = [text.get(w, 0) for w, _ in top_words_160]

print(data[:5])
