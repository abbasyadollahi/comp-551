import time
from sklearn.model_selection import train_test_split, GridSearchCV

from data import load_data
from pipeline import linear_svc_pipeline

def execute_pipeline(title, pipeline, x, y, x_v, y_v):
    print(title)
    start = time.time()
    pipeline.fit(x, y)
    train_time = time.time()
    print(f'Training time: {train_time-start}')
    print(f'Training accuracy: {pipeline.score(x, y)}')
    print(f'Validation accuracy: {pipeline.score(x_v, y_v)}')
    print(f'Scoring time: {time.time()-train_time}s')
    print('')

if __name__ == '__main__':
    print('Loading data...')
    start = time.time()
    mr_data, mr_labels = load_data('MR.pkl')
    mr_train_data, mr_dev_data, mr_train_labels, mr_dev_labels = train_test_split(mr_data, mr_labels, test_size=0.3, random_state=42)
    sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_data('SST2.pkl')
    print(f'Time to load data: {time.time()-start}s')

    # pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    # execute_pipeline('Linear SVC - MR Dataset', pipeline, mr_train_data, mr_train_labels, mr_dev_data, mr_dev_labels)
    # execute_pipeline('Linear SVC - SST Dataset', pipeline, sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels)

    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__tol': [1e-4, 1e-5, 1e-6],
        'clf__max_iter': [500, 1000, 2000]
    }

    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    cv = GridSearchCV(pipeline, param_grid, iid=False, cv=5, verbose=1, n_jobs=-1)
    cv.fit(mr_train_data, mr_train_labels)
    print("Best parameter (CV score=%0.3f):" % cv.best_score_)
    print(cv.best_params_)

    pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    cv = GridSearchCV(pipeline, param_grid, iid=False, cv=5, verbose=1, n_jobs=-1)
    cv.fit(sst_train_data, sst_train_labels)
    print("Best parameter (CV score=%0.3f):" % cv.best_score_)
    print(cv.best_params_)
