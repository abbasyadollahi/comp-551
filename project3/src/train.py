# -*- coding: utf-8 -*-
import keras
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta
from data import RAW_DIR, get_dataset, get_csv

def main():
    logger = logging.getLogger(__name__)

    num_classes = 10
    batch_size = 512 # best 512
    num_steps = 4000 # best 2000
    epochs = 40

    img_x, img_y = 64, 64

    X, y = get_dataset(RAW_DIR / 'train_images.pkl'), \
            get_csv(RAW_DIR / 'train_labels.csv')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    main()
