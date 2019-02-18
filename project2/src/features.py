import sklearn
from sklearn.datasets import load_files

moviedir = r'F:\\MENG_FILES\\Winter_2019\\551_Applied_Machine_Learning\\Mini_project_2\\comp_551_imbd_sentiment_classification\\'
train_dir = moviedir + "train\\train\\"
test_dir = moviedir + "test\\"

# loading all files as training data. 
movie_train = load_files(train_dir, shuffle=False)

# Loading all files as test data.
movie_test = load_files(test_dir, shuffle=False)

'''
REMOVE_STOPWORDS - takes a sentence and the stopwords as inputs and returns the sentence without any stopwords 
Sentence - The input from which the stopwords have to be removed
Stopwords - A list of stopwords 

'''
def remove_stopwords(sentence, stopwords):
    sentencewords = sentence.split()
    resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result

def imdb_data_preprocess(inpath, outpath="./", mix=False):
    import pandas as pd 
    from pandas import DataFrame, read_csv
    import os
    import csv 
    import numpy as np 

    stopwords = open(inpath+"stopwords.txt", 'r', encoding="ISO-8859-1").read()
    stopwords = stopwords.split("\n")

    indices = []
    text = []
    rating = []

    i =  0 

    for filename in os.listdir(inpath+"pos"):
        data = open(train_dir+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        rating.append("1")
        i = i + 1

    for filename in os.listdir(inpath+"neg"):
        data = open(train_dir+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        rating.append("0")
        i = i + 1

    Dataset = list(zip(indices,text,rating))

    if mix:
        np.random.shuffle(Dataset)

    df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])

    return df

'''
UNIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the unigram as output 
Data - The data for which the unigram model has to be fit 

'''
def unigram_process(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(data)
    return vectorizer


'''
BIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the bigram as output 
Data - The data for which the bigram model has to be fit 

'''
def bigram_process(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1,2))
    vectorizer = vectorizer.fit(data)
    return vectorizer


'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output 
Data - The data for which the bigram model has to be fit 

'''
def tfidf_process(data):
    from sklearn.feature_extraction.text import TfidfTransformer 
    transformer = TfidfTransformer()
    transformer = transformer.fit(data)
    return transformer


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data is returned 

'''
def retrieve_data(data, train=True):
    import pandas as pd 
    X = data['text']

    if train:
        Y = data['polarity']
        return X, Y

    return X

data_df = imdb_data_preprocess(inpath=train_dir)
data_df.head()

X, y = retrieve_data(data_df)

uf = unigram_process(X)

bf = bigram_process(X)

uni_tfidf = tfidf_process(uf)

bi_tfidf = tfidf_process(bf)

