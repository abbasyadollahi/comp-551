import time
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
from copy import deepcopy

from features import PreprocessData
from linear_regression import linear_closed_form, linear_gradient_descent

ppd = PreprocessData()
train, validation, test = ppd.preprocess_data(ppd.data)
ppd.compute_most_common_words(train, regex=True)

# Baseline - no extra features
# Compute features on training set
start = time.time()
X_train, y_train = ppd.compute_features(train, extra_features=False, num_word_features=0)
feat_train_runtime = time.time() - start
print(X_train.shape)

# Compute features on validation set
start = time.time()
X_valid, y_valid = ppd.compute_features(validation, extra_features=False, num_word_features=0)
feat_valid_runtime = time.time() - start
print(X_valid.shape)

print(f'Training features runtime: {feat_train_runtime}')
print(f'Validation features runtime: {feat_valid_runtime}')

# w_init = np.zeros(X_train.shape[1])
# w_grad = linear_gradient_descent(X_train, y_train, w_init, decay_speed=10**(-10), learn_rate=10**(-8), min_err=10**(-7), max_iter=10000000, verbose=True)
w_grad = linear_closed_form(X_train, y_train)

# Compute MSE on validation set
y_grad_valid = np.matmul(X_valid, w_grad)
mse_grad_valid = np.sum((y_grad_valid - y_valid)**2)/len(y_valid)
print(mse_grad_valid)


# Extra features
# Compute features on training set
start = time.time()
X_train_extra, y_train_extra = ppd.compute_features(train, extra_features=True, num_word_features=0)
feat_train_runtime = time.time() - start
print(X_train_extra.shape)

# Compute features on validation set
start = time.time()
X_valid_extra, y_valid_extra = ppd.compute_features(validation, extra_features=True, num_word_features=0)
feat_valid_runtime = time.time() - start
print(X_valid_extra.shape)

print(f'Training features runtime: {feat_train_runtime}')
print(f'Validation features runtime: {feat_valid_runtime}')

# w_init_extra = np.zeros(X_train_extra.shape[1])
# w_grad_extra = linear_gradient_descent(X_train_extra, y_train_extra, w_init_extra, decay_speed=10**(-10), learn_rate=10**(-8), min_err=10**(-7), max_iter=10000000, verbose=True)
w_grad_extra = linear_closed_form(X_train_extra, y_train_extra)

# Compute MSE on validation set
y_grad_valid_extra = np.matmul(X_valid_extra, w_grad_extra)
mse_grad_valid_extra = np.sum((y_grad_valid_extra - y_valid_extra)**2)/len(y_valid_extra)
print(mse_grad_valid_extra)

print(mse_grad_valid - mse_grad_valid_extra)