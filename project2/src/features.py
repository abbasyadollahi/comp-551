import os
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# TODO: Implement two feature extraction pipelines. (Use lemmatization, tokenizing, etc.)

def load_train(data_path):
    data = []
    for category in os.listdir(data_path):
        category_bin = 0 if category == 'neg' else 1
        for file_name in os.listdir(os.path.join(data_path, category)):
            with open(os.path.join(data_path, category, file_name), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').strip().lower()
                data.append([review, category_bin])
    return data

class LemmaTokenizer(object):
    '''
    Tokenizes, lemmatizes and removes stop words.
    Pass this as tokenizer for CountVectorizer.
    '''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        tokens_no_stop = [t for t in tokens if t not in self.stop_words]
        return [self.wnl.lemmatize(t) for t in tokens_no_stop]

try:
    os.chdir(os.path.join(os.getcwd(), 'project2/src'))
    print(os.getcwd())
except:
    pass

data_dir = '../data'
train_dir = os.path.join(data_dir, 'train')

# loading all files as training data. 
data_train = load_train(train_dir)
train, validation = train_test_split(data_train[:100], test_size=0.2)
print(f'Test set length: {len(train)}')
print(f'Validation set length: {len(validation)}')

binary_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1,1), binary=True)
unigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2), norm='l2')

X_binary = binary_vectorizer.fit_transform([data[0] for data in train]).toarray()
X_unigram = unigram_vectorizer.fit_transform([data[0] for data in train]).toarray()
X_bigram = bigram_vectorizer.fit_transform([data[0] for data in train]).toarray()
X_tfidf = tfidf_vectorizer.fit_transform([data[0] for data in train]).toarray()
y = np.array([data[1] for data in train])

print(X_binary.shape)
print(X_unigram.shape)
print(X_bigram.shape)
print(X_tfidf.shape)
print(y.shape)
