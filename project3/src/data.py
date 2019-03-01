import numpy as np
import pandas as pd
import os

DATA_DIR = './project3/data'

def get_train_data():
    train_images = pd.read_pickle(os.path.join(DATA_DIR, 'train_images.pkl'))
    train_labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
    return train_images, train_labels

def get_test_data():
    return pd.read_pickle(os.path.join(DATA_DIR, 'test_images.pkl'))
