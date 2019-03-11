import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'data'))

def load_train():
    train_images = pd.read_pickle(op.join(DATA_DIR, 'train_images.pkl'))
    train_labels = pd.read_csv(op.join(DATA_DIR, 'train_labels.csv')).to_numpy()[:, 1]
    return train_images, train_labels

def load_test():
    return pd.read_pickle(op.join(DATA_DIR, 'test_images.pkl'))

# train, csv = load_train()
# test = load_test()
# import cv2
# cv2.imwrite('gang.jpg', train[0])
# plt.imshow(train[0], cmap='gray')
# plt.show()
