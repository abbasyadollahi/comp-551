import time
from sklearn.model_selection import train_test_split

from data import load_train, load_test, predictions_to_csv, save_model, load_model
from pipeline import naive_bayes_pipeline, log_reg_pipeline, linear_svc_pipeline

if __name__ == '__main__':
    # Loading all files as training data
    print('Loading data...')
    start = time.time()
    data_train = load_train()
    train, validation = train_test_split(data_train, test_size=0.25)

    x, y = zip(*train)
    x_v, y_v = zip(*validation)
    print(f'Time to load data: {time.time()-start}')


    # print('#### Training Logistic Regression ####')
    # print('1. Unigram')
    # pipeline = log_reg_pipeline()

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'logreg_unigram.joblib')

    # print('2. Bigram')
    # pipeline = log_reg_pipeline(bigram=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'logreg_bigram.joblib')

    # print('3. Unigram + TFIDF')
    # pipeline = log_reg_pipeline(tfidf=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'logreg_unigram_tfidf.joblib')

    # print('4. Bigram + TFIDF')
    # pipeline = log_reg_pipeline(bigram=True, tfidf=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'logreg_bigram_tfidf.joblib')


    # print('#### Training Linear SVC ####')
    # print('1. Unigram')
    # pipeline = linear_svc_pipeline()

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'linsvc_unigram.joblib')

    # print('2. Bigram')
    # pipeline = linear_svc_pipeline(bigram=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'linsvc_bigram.joblib')

    # print('3. Unigram + TFIDF')
    # pipeline = linear_svc_pipeline(tfidf=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'linsvc_unigram_tfidf.joblib')

    # print('4. Bigram + TFIDF')
    # pipeline = linear_svc_pipeline(bigram=True, tfidf=True)

    # start = time.time()
    # pipeline.fit(x, y)
    # print(f'Training time: {time.time()-start}')
    # print(f'Training accuracy: {pipeline.score(x, y)}')
    # print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # save_model(pipeline, 'linsvc_bigram_tfidf.joblib')


    print('#### Training Naive Bayes ####')
    print('1. Unigram')
    pipeline = naive_bayes_pipeline()

    start = time.time()
    pipeline.fit(x[:100], y[:100])
    print(f'Training time: {time.time()-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    save_model(pipeline, 'nb_unigram.joblib')
