from pandas import DataFrame
import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
RESULT_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'results'))

def load_train():
    train_images = pd.read_pickle(op.join(DATA_DIR, 'train_images.pkl'))
    train_labels = pd.read_csv(op.join(DATA_DIR, 'train_labels.csv')).to_numpy()[:, 1]
    return train_images, train_labels

def load_test():
    return pd.read_pickle(op.join(DATA_DIR, 'test_images.pkl'))

def predictions_to_csv(pred, filename):
    if not op.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    df = DataFrame(pred)
    df.to_csv(op.join(RESULT_DIR, filename), index_label=['Id', 'Category'])
