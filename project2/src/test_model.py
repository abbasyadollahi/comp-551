import time

from data import load_test, predictions_to_csv, load_model

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    data_test = load_test()
    print(f'Time to load data: {time.time()-start}')

    # Replace with name of model you want load
    pipeline = load_model('sgd_bigram_tfidf.joblib')

    # Generate predictions
    pred = pipeline.predict(data_test)

    # Name of csv to save predictions in
    predictions_to_csv(pred, 'sgd_bigram_tfidf.csv')
