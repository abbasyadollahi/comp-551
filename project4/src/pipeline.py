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
        # and abs(self.sentiment_strength.get(t, 1)) > 0.4]
        tokens_no_stop = [t for t in tokens if t not in self.stop_words]
        return [self.wnl.lemmatize(t) for t in tokens_no_stop]
        # return [self.wnl.lemmatize(t, pos=self.penn_to_wn(tag)) for t, tag in pos_tag(tokens_no_stop)]

    def penn_to_wn(self, tag):
        tags = {
            'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ,
            'RB': wn.ADV, 'RBS': wn.ADV, 'RBR': wn.ADV,
            'NN': wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
            'VB': wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
        }
        return tags.get(tag, wn.NOUN)

def get_vectorizer(max_features, binary=False, bigram=False, tfidf=False):
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), max_df=0.5, max_features=max_features)
    if tfidf:
        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_df=0.5, max_features=max_features, norm='l2')
    if binary:
        vectorizer.set_params(binary=True)
    if bigram:
        vectorizer.set_params(ngram_range=(1, 2))
    return vectorizer
