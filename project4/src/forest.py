import time
import numpy as np
from gcforest.gcforest import GCForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from data import load_mr, load_sst2
from pipeline import get_vectorizer

config = {}
cascade = {
    'random_state': 0,
    'early_stopping_rounds': 4,
    'n_classes': 2,
    'estimators': [
        {'n_folds': 3, 'type': 'ExtraTreesClassifier', 'n_estimators': 500,
            'max_depth': None, 'n_jobs': -1, 'max_features': 1},
        {'n_folds': 3, 'type': 'ExtraTreesClassifier', 'n_estimators': 500,
            'max_depth': None, 'n_jobs': -1, 'max_features': 1},
        {'n_folds': 3, 'type': 'ExtraTreesClassifier', 'n_estimators': 500,
            'max_depth': None, 'n_jobs': -1, 'max_features': 1},
        {'n_folds': 3, 'type': 'ExtraTreesClassifier', 'n_estimators': 500,
            'max_depth': None, 'n_jobs': -1, 'max_features': 1},
        {'n_folds': 3, 'type': 'RandomForestClassifier',
            'n_estimators': 500, 'max_depth': None, 'n_jobs': -1},
        {'n_folds': 3, 'type': 'RandomForestClassifier',
            'n_estimators': 500, 'max_depth': None, 'n_jobs': -1},
        {'n_folds': 3, 'type': 'RandomForestClassifier',
            'n_estimators': 500, 'max_depth': None, 'n_jobs': -1},
        {'n_folds': 3, 'type': 'RandomForestClassifier',
            'n_estimators': 500, 'max_depth': None, 'n_jobs': -1}
    ]
}
config['cascade'] = cascade

print('Loading data...')
start = time.time()
mr_data, mr_labels = load_mr()
mr_data, mr_labels = np.asarray(mr_data), np.asarray(mr_labels)
sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = load_sst2()
sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels = np.asarray(sst_train_data), np.asarray(sst_train_labels), np.asarray(sst_dev_data), np.asarray(sst_dev_labels), np.asarray(sst_test_data), np.asarray(sst_test_labels)
print(f'Time to load data: {time.time()-start}s')

# vect = get_vectorizer(None, ngram=1, tfidf=True)
# gc = GCForest(config)

# X_train_mr = vect.fit_transform(mr_data)
# print('GCForest - MR Dataset')
# scores = cross_val_score(gc, X_train_mr, mr_labels, cv=10, n_jobs=-1)
# print(f'Training Accuracy: {scores.mean()}')
# mr_pred = cross_val_predict(gc, X_train_mr, mr_labels, cv=10, n_jobs=-1)
# print(f'Validation Accuracy: {accuracy_score(mr_labels, mr_pred)}')

vect = get_vectorizer(None, ngram=1, tfidf=True)
gc = GCForest(config)

X_train_sst = vect.fit_transform(sst_train_data)
print(X_train_sst.shape)
print('GCForest - SST Dataset')
gc.fit_transform(X_train_sst, sst_train_labels)
sst_pred = gc.predict(X_train_sst)
print(f'Training Accuracy: {accuracy_score(sst_train_labels, sst_pred)}')
X_dev_sst = vect.fit_transform(sst_dev_data)
sst_pred = gc.predict(X_dev_sst)
print(f'Validattion Accuracy: {accuracy_score(sst_dev_labels, sst_pred)}')
