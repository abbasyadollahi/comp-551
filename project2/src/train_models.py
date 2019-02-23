import time
from sklearn.model_selection import train_test_split

from data import load_train, load_test, predictions_to_csv, save_model, load_model
from pipeline import naive_bayes_pipeline, log_reg_pipeline, linear_svc_pipeline, sgd_pipeline

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
    print('#### Training Logistic Regression ####')
    print('1. Unigram + TFIDF')
    pipeline = log_reg_pipeline(max_features, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'logreg_unigram_tfidf.joblib')

    print('2. Bigram + TFIDF')
    pipeline = log_reg_pipeline(max_features, bigram=True, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'logreg_bigram_tfidf.joblib')

    print('#### Training SGD Classifier ####')
    print('1. Unigram + TFIDF')
    pipeline = sgd_pipeline(max_features, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'sgd_unigram_tfidf.joblib')

    print('2. Bigram + TFIDF')
    pipeline = sgd_pipeline(max_features, bigram=True, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'sgd_bigram_tfidf.joblib')

    print('#### Training Linear SVC ####')
    print('1. Unigram + TFIDF')
    pipeline = linear_svc_pipeline(max_features, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'linsvc_unigram_tfidf.joblib')

    print('2. Bigram + TFIDF')
    pipeline = linear_svc_pipeline(max_features, bigram=True, tfidf=True)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'linsvc_bigram_tfidf.joblib')

    print('#### Training Naive Bayes ####')
    print('1. Unigram')
    pipeline = naive_bayes_pipeline(max_features=900)

    start = time.time()
    pipeline.fit(x, y)
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'nb_unigram.joblib')
