from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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
        return tokens
        # return [self.wnl.lemmatize(t) for t in tokens]

def get_vectorizer(max_features, binary=False, ngram=1, tfidf=False):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, ngram_range=(1, ngram))
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, norm='l2')
    if binary:
        vectorizer.set_params(binary=True)
    return vectorizer

def linear_svc_pipeline(max_features, ngram=1, tfidf=False):
    vectorizer = get_vectorizer(max_features, ngram=ngram, tfidf=tfidf)
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', LinearSVC(loss='squared_hinge', C=1, tol=1e-4, max_iter=1000))
    ])
    return pipeline

# def nbsvm_pipeline(max_features, ngram=1, tfidf=False):
#     vectorizer = get_vectorizer(max_features, binary=True, ngram=ngram, tfidf=tfidf)
#     nb = NaiveBayes(vectorizer)
#     return nb
