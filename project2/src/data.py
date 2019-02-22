import re
import os
import os.path as op
from pandas import DataFrame, read_csv
from joblib import dump, load

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
TEST_DIR = op.join(DATA_DIR, 'test')
TRAIN_DIR = op.join(DATA_DIR, 'train')
MODEL_DIR = op.join(DATA_DIR, 'model')
RESULT_DIR = op.join(DATA_DIR, 'result')

def load_test():
    data = []
    for filename in sorted(os.listdir(TEST_DIR), key=lambda x: int(x.split('.')[0])):
        with open(op.join(TEST_DIR, filename), 'rb') as f:
            review = clean_review(f.read().decode('utf-8').lower())
            data.append(review)
    return data

def load_train():
    data = []
    for category in os.listdir(TRAIN_DIR):
        category_bin = 1 if category == 'pos' else 0
        for filename in os.listdir(op.join(TRAIN_DIR, category)):
            with open(op.join(TRAIN_DIR, category, filename), 'rb') as f:
                review = clean_review(f.read().decode('utf-8').lower())
                data.append([review, category_bin])
    return data

def clean_review(review):
    replacements = {
        '<br />': ' ', "'m": ' am', "'s": ' is', "'t": ' not',
        "'d": ' would', "'re": ' are', "'ll": ' will', "'ve": ' have'
    }

    regexp = re.compile('|'.join(map(re.escape, replacements)))
    return regexp.sub(lambda match: replacements[match.group(0)], review)

def load_sentiment_strength():
    return read_csv(op.join(DATA_DIR, 'sentiment_strength.csv'), index_col=0, squeeze=True).to_dict()

def predictions_to_csv(pred, filename):
    if not op.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    df = DataFrame(pred)
    df.to_csv(op.join(RESULT_DIR, filename), index_label=['Id', 'Category'])

def save_model(model, filename):
    '''
    model: Trained classifier
    filename: model.joblib
    '''
    if not op.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    dump(model, op.join(MODEL_DIR, filename))

def load_model(filename):
    '''
    filename: model.joblib
    '''
    return load(op.join(MODEL_DIR, filename))
