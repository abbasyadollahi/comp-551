import pickle
import numpy as np
import pandas as pd
import os.path as op
from keras import backend as K
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))
MR_FILE = op.join(DATA_DIR, 'mr')

def load_mr_data():
    return pd.read_pickle(MR_FILE)
