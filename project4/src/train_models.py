import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score

from data import load_mr, load_sst2
from pipeline import linear_svc_pipeline, nbsvm_pipeline

def execute_pipeline(title, pipeline, x, y, x_v, y_v, x_t, y_t):
    print(title)
    start = time.time()
    pipeline.fit(x, y)
    train_time = time.time()
    print(f'Training time: {train_time-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    print(f'Test accuracy: {pipeline.score(x_t, y_t)}')
    print(f'Scoring time: {time.time()-train_time}s')

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    mr_data, mr_labels = load_mr()
    sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_sst2()
    print(f'Time to load data: {time.time()-start}s')

    print('##### Training Linear SVC #####')
    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    execute_pipeline('Linear SVC - SST Dataset', pipeline, sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels)

    print('Linear SVC - MR Dataset')
    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    scores = cross_val_score(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    print(f'Training Accuracy: {scores.mean()}')
    mr_pred = cross_val_predict(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    print(f'Validation Accuracy: {accuracy_score(mr_labels, mr_pred)}')


    print('##### Training Naive Bayes SVM #####')
    # mr_labels = [int(x) for x in mr_labels]
    print(type(mr_data))
    print(type(mr_data[0]))
    print(mr_data.shape)
    print(type(mr_labels))
    print(type(mr_labels[0]))
    print(mr_labels.shape)

    model, x_train = nbsvm_pipeline(mr_data, mr_labels, max_features=10000, ngram=3, tfidf=False)
    xt, xv, yt, yv = train_test_split(x_train, mr_labels, test_size=0.3, random_state=42)
    model.fit(xt, yt, batch_size=32, epochs=3, validation_data=(xv, yv))
    # execute_pipeline('1. NBSVM', pipeline, 'nbsvm')
