from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet as wn
from nltk import pos_tag

class LemmaTokenizer:
    '''
    Tokenizes, lemmatizes and removes stop words.
    Pass this as tokenizer for CountVectorizer.
    '''

    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['&', "'", "''", '``', '(', ')', ',', ':', ';'])

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        # tokens_no_stop = [t for t in tokens if t not in self.stop_words]
        # return tokens
        return [self.wnl.lemmatize(t) for t in tokens]

def get_vectorizer(max_features, binary=False, bigram=False, trigram=False, tfidf=False):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features)
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, norm='l2')
    if binary:
        vectorizer.set_params(binary=True)
    if bigram:
        vectorizer.set_params(ngram_range=(1, 2))
    if trigram:
        vectorizer.set_params(ngram_range=(1, 3))
    return vectorizer

def linear_svc_pipeline(max_features, bigram=False, trigram=False, tfidf=False):
    vectorizer = get_vectorizer(max_features, bigram=bigram, tfidf=tfidf)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', LinearSVC(C=1, tol=1e-4, max_iter=1000))
    ])
    return pipeline
