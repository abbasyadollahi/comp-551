import os
import time
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_files
from keras.utils import plot_model
from matplotlib import pyplot as plt

from data import load_mr, load_sst2
from nbsvm import get_nbsvm_model
from pipeline import linear_svc_pipeline, nbsvm_pipeline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

    # print('##### Training Linear SVC #####')
    # pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    # execute_pipeline('Linear SVC - SST Dataset', pipeline, sst_train_data, sst_train_labels, sst_dev_data, sst_dev_labels, sst_test_data, sst_test_labels)

    # print('Linear SVC - MR Dataset')
    # pipeline = linear_svc_pipeline(max_features=None, ngram=2, tfidf=True)
    # scores = cross_val_score(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    # print(f'Training Accuracy: {scores.mean()}')
    # mr_pred = cross_val_predict(pipeline, mr_data, mr_labels, cv=10, n_jobs=-1)
    # print(f'Validation Accuracy: {accuracy_score(mr_labels, mr_pred)}')


    print('##### Training Naive Bayes SVM #####')
    print('Naive Bayes SVM - MR Dataset')
    cvscores = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    x_mr, nb_ratios, num_words = nbsvm_pipeline(mr_data, mr_labels, max_features=100000, ngram=3, tfidf=False)
    for trn, tst in kfold.split(mr_data, mr_labels):
        model = get_nbsvm_model(num_words, nb_ratios=nb_ratios)
        model.fit(x_mr[trn], mr_labels[trn], validation_data=(x_mr[tst], mr_labels[tst]), batch_size=16, epochs=5, verbose=0)

        score = model.evaluate(x_mr[tst], mr_labels[tst], verbose=0)[1]
        cvscores.append(score)
        print(f'Acc: {score:.2f}')

    print(f'Avg: {np.mean(cvscores):.2f} (+/- {np.std(cvscores):.2f})')

    print('Naive Bayes SVM - SST2 Dataset')
    sst_train_len, sst_dev_len, sst_test_len = map(len, [sst_train_data, sst_dev_data, sst_test_data])
    sst_data = np.concatenate([sst_train_data, sst_dev_data, sst_test_data])
    sst_labels = np.concatenate([sst_train_labels, sst_dev_labels, sst_test_labels])
    x_sst, nb_ratios, num_words = nbsvm_pipeline(sst_data, sst_labels, max_features=100000, ngram=3, tfidf=False)
    x_train, y_train = x_sst[:sst_train_len], sst_labels[:sst_train_len]
    x_dev, y_dev = x_sst[sst_train_len:sst_train_len+sst_dev_len], sst_labels[sst_train_len:sst_train_len+sst_dev_len]
    x_test, y_test = x_sst[sst_train_len+sst_dev_len:sst_train_len+sst_dev_len+sst_test_len], sst_labels[sst_train_len+sst_dev_len:sst_train_len+sst_dev_len+sst_test_len]
    model = get_nbsvm_model(num_words, nb_ratios=nb_ratios)
    history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=16, epochs=5)
    _, accuracy = model.evaluate(x_test, y_test)
    print(accuracy)
    plot_model(model, to_file='./project4/nbsvm_arch.png', show_shapes=True)

    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epcs = range(1, len(acc) + 1)
    plt.plot(epcs, acc, 'bo', label='Training acc')
    plt.plot(epcs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # x_mr, nb_ratios, num_words = nbsvm_pipeline(mr_data, mr_labels, max_features=100000, ngram=3, tfidf=False)
    # xt, xv, yt, yv = train_test_split(x_mr, mr_labels, test_size=0.3, random_state=42)
    # model = get_nbsvm_model(num_words, nb_ratios=nb_ratios)
    # model.fit(xt, yt, validation_data=(xv, yv), batch_size=5, epochs=16)
