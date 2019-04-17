import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from data import load_data
from pipeline import linear_svc_pipeline

def execute_pipeline(title, pipeline, x, y, x_v, y_v, x_t, y_t):
    print(title)
    start = time.time()
    pipeline.fit(x, y)
    train_time = time.time()
    print(f'Training time: {train_time-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    # print(f'Test accuracy: {pipeline.score(x_t, y_t)}')
    print(f'Scoring time: {time.time()-train_time}s')

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    mr_data, mr_labels = load_data('MR.pkl')
    sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_data('SST2.pkl')
    print(f'Time to load data: {time.time()-start}s')

    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    execute_pipeline('Linear SVC - SST Dataset', pipeline, sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels)

    print('Linear SVC - MR Dataset')
    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    scores = cross_val_score(pipeline, mr_data, mr_labels, cv=5, n_jobs=-1)
    print(f'Training Accuracy: {scores.mean()}')
    mr_pred = cross_val_predict(pipeline, mr_data, mr_labels, cv=5, n_jobs=-1)
    print(f'Validation accuracy: {accuracy_score(mr_labels, mr_pred)}')
