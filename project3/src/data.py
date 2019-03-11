# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#import os
from pathlib import Path

#DATA_DIR = './project3/data'

project_dir = Path(__file__).resolve().parents[1]
raw_path = project_dir / 'data/raw'

def get_train_data():
    train_images = pd.read_pickle(raw_path / 'train_images.pkl')
    train_labels = pd.read_csv(raw_path / 'train_labels.csv')
    return train_images, train_labels

def get_test_data():
    return pd.read_pickle(raw_path / 'test_images.pkl')
