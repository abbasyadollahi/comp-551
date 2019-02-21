import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from data import load_test, load_train


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


# Loading all files as training data
data_train = load_train()
train, validation = train_test_split(data_train, test_size=0.2)

binary_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), binary=True)
unigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2), norm='l2')

x, y = zip(*train)
X_binary = binary_vectorizer.fit_transform(x).toarray()
X_unigram = unigram_vectorizer.fit_transform(x).toarray()
X_bigram = bigram_vectorizer.fit_transform(x).toarray()
X_tfidf = tfidf_vectorizer.fit_transform(x).toarray()
y = np.array(y)

# print(f'Test set length: {len(train)}')
# print(f'Validation set length: {len(validation)}')
# print(X_binary.shape)
# print(X_unigram.shape)
# print(X_bigram.shape)
# print(X_tfidf.shape)
# print(y.shape)

def binary_vector():
    # binary_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), binary=True)
    # X_binary = binary_vectorizer.fit_transform(x).toarray()
    return X_binary

def unigram_vector():
    # unigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1))
    # X_unigram = unigram_vectorizer.fit_transform(x).toarray()
    return X_unigram

def bigram_vector():
    # bigram_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
    # X_bigram = bigram_vectorizer.fit_transform(x).toarray()
    return X_bigram

def tfidf_vector():
    # tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2), norm='l2')
    # X_tfidf = tfidf_vectorizer.fit_transform(x).toarray()
    return X_tfidf

def y_vector():
    # y = np.array(y)
    return y
