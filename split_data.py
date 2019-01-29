import re
import json
from collections import Counter

regex = "[^\w'_]+"

with open('proj1_data.json') as f:
    data = json.load(f)

word_count = Counter()
for dp in data[:10000]:
    text = dp['text'].lower()
    dp['text'] = text.split()
    dp['regex'] = re.split(regex, text)
    word_count += Counter(dp['text'])
for dp in data[10000:]:
    text = dp['text'].lower()
    dp['text'] = text.split()
    dp['regex'] = re.split(regex, text)

train_data = data[:10000]
test_data = data[10000:11000]
validation_data = data[11000:]

top_words_160 = word_count.most_common(160)
for dp in data:
    text = Counter(dp['text'])
    dp['x160'] = [text.get(w, 0) for w, _ in top_words_160]

print(data[:5])
