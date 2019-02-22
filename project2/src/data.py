import os
import os.path as op
from pandas import DataFrame
from joblib import dump, load

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
TEST_DIR = op.join(DATA_DIR, 'test')
TRAIN_DIR = op.join(DATA_DIR, 'train')
MODEL_DIR = op.join(DATA_DIR, 'model')
RESULT_DIR = op.join(DATA_DIR, 'result')

def load_test():
    data = []
    for filename in sorted(os.listdir(TEST_DIR), key=lambda x: int(x.split('.')[0])):
        with open(os.path.join(TEST_DIR, filename), 'rb') as f:
            review = f.read().decode('utf-8').replace('<br />', ' ').lower()
            data.append(review)
    return data

def load_train():
    data = []
    for category in os.listdir(TRAIN_DIR):
        category_bin = 1 if category == 'pos' else 0
        for filename in os.listdir(os.path.join(TRAIN_DIR, category)):
            with open(os.path.join(TRAIN_DIR, category, filename), 'rb') as f:
                review = f.read().decode('utf-8').replace('<br />', ' ').lower()
                data.append([review, category_bin])
    return data

def predictions_to_csv(pred, filename):
    df = DataFrame(pred)
    df.to_csv(op.join(RESULT_DIR, filename), index_label=['Id', 'Category'])

def save_model(model, filename):
    '''
    model: Trained classifier
    filename: model.joblib
    '''
    dump(model, op.join(MODEL_DIR, filename))

def load_model(filename):
    '''
    filename: model.joblib
    '''
    return load(op.join(MODEL_DIR, filename))
