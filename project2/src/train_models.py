import time
from sklearn.model_selection import train_test_split

from data import load_train, save_model
from pipeline import naive_bayes_pipeline, log_reg_pipeline, linear_svc_pipeline, sgd_pipeline

def execute_pipeline(title, pipeline, model_name):
    print(title)
    start = time.time()
    pipeline.fit(x, y)
    train_time = time.time()
    print(f'Training time: {train_time-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    print(f'Scoring time: {time.time()-train_time}')
    save_model(pipeline, f'{model_name}.joblib')

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    max_features = None
    data_train = load_train()
    train, validation = train_test_split(data_train, test_size=0.3, random_state=42)

    x, y = zip(*train)
    x_v, y_v = zip(*validation)
    print(f'Time to load data: {time.time()-start}')
    print(f'Training with max_features: {max_features}')

    logreg_unigram_tfidf = logreg_bigram_tfidf = sgd_unigram_tfidf = sgd_bigram_tfidf = linsvc_unigram_tfidf = linsvc_bigram_tfidf = nb_unigram = True
    # logreg_unigram_tfidf = False
    # logreg_bigram_tfidf = False
    # sgd_unigram_tfidf = False
    # sgd_bigram_tfidf = False
    # linsvc_unigram_tfidf = False
    # linsvc_bigram_tfidf = False
    # nb_unigram = False

    print('##### Training Logistic Regression #####')
    if logreg_unigram_tfidf:
        pipeline = log_reg_pipeline(max_features, tfidf=True)
        execute_pipeline('1. Unigram + TFIDF', pipeline, 'logreg_unigram_tfidf')
    if logreg_bigram_tfidf:
        pipeline = log_reg_pipeline(max_features, bigram=True, tfidf=True)
        execute_pipeline('2. Bigram + TFIDF', pipeline, 'logreg_bigram_tfidf')

    print('##### Training SGD Classifier #####')
    if sgd_unigram_tfidf:
        pipeline = sgd_pipeline(max_features, tfidf=True)
        execute_pipeline('1. Unigram + TFIDF', pipeline, 'sgd_unigram_tfidf')
    if sgd_bigram_tfidf:
        pipeline = sgd_pipeline(max_features, bigram=True, tfidf=True)
        execute_pipeline('2. Bigram + TFIDF', pipeline, 'sgd_bigram_tfidf')

    print('##### Training Linear SVC #####')
    if linsvc_unigram_tfidf:
        pipeline = linear_svc_pipeline(max_features, tfidf=True)
        execute_pipeline('1. Unigram + TFIDF', pipeline, 'linsvc_unigram_tfidf')
    if linsvc_bigram_tfidf:
        pipeline = linear_svc_pipeline(max_features, bigram=True, tfidf=True)
        execute_pipeline('2. Bigram + TFIDF', pipeline, 'linsvc_bigram_tfidf')

    print('##### Training Naive Bayes #####')
    if nb_unigram:
        pipeline = naive_bayes_pipeline(max_features=900)
        execute_pipeline('1. Unigram', pipeline, 'nb_unigram')
