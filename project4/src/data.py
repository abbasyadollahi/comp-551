import os
import os.path as op
from pandas import DataFrame, read_pickle

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))

def load_data(filename):
    df = read_pickle(op.join(DATA_DIR, filename))
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    for idx, row in df.iterrows():
        if row['split'] == 'train':
            train_data.append(row['sentence'])
            train_labels.append(row['label'])
        elif row['split'] == 'dev':
            dev_data.append(row['sentence'])
            dev_labels.append(row['label'])
        elif row['split'] == 'test':
            test_data.append(row['sentence'])
            test_labels.append(row['label'])
        else:
            raise Exception(f'Unknown label {row["split"]}')
    if len(dev_data) == 0 or len(test_data) == 0:
        return train_data, train_labels
    else:
        return train_data, train_labels, dev_data, dev_labels, test_data, test_labels
