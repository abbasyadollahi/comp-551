import time
import numpy as np

from features import PreprocessData
from linear_regression import linear_closed_form, linear_gradient_descent

ppd = PreprocessData()
train, validation, test = ppd.preprocess_data(ppd.data)
ppd.compute_most_common_words(train, regex=True)
num_words = 60

# Baseline - no extra features
print('Testing baseline')
# Compute features on training set
start = time.time()
X_train, y_train = ppd.compute_features(train, extra_features=False, num_word_features=num_words)
feat_train_runtime = time.time() - start
print(X_train.shape)

# Compute features on validation set
start = time.time()
X_valid, y_valid = ppd.compute_features(validation, extra_features=False, num_word_features=num_words)
feat_valid_runtime = time.time() - start
print(X_valid.shape)

print(f'Training features runtime: {feat_train_runtime}')
print(f'Validation features runtime: {feat_valid_runtime}')

# w_init = np.zeros(X_train.shape[1])
# w_grad = linear_gradient_descent(X_train, y_train, w_init, decay_speed=10**(-10), learn_rate=10**(-8), min_err=10**(-7), max_iter=10000000, verbose=True)
w_closed = linear_closed_form(X_train, y_train)

# Compute MSE on training set
y_grad_train = np.matmul(X_train, w_closed)
mse_grad_train = np.sum((y_grad_train - y_train)**2)/len(y_train)
print(f'MSE Training: {mse_grad_train}')

# Compute MSE on validation set
y_grad_valid = np.matmul(X_valid, w_closed)
mse_grad_valid = np.sum((y_grad_valid - y_valid)**2)/len(y_valid)
print(f'MSE Validation: {mse_grad_valid}')


# Extra features
print('Testing extra features')
# Compute features on training set
start = time.time()
X_train_extra, y_train_extra = ppd.compute_features(train, extra_features=True, num_word_features=num_words)
feat_train_runtime = time.time() - start
print(X_train_extra.shape)

# Compute features on validation set
start = time.time()
X_valid_extra, y_valid_extra = ppd.compute_features(validation, extra_features=True, num_word_features=num_words)
feat_valid_runtime = time.time() - start
print(X_valid_extra.shape)

print(f'Training features runtime: {feat_train_runtime}')
print(f'Validation features runtime: {feat_valid_runtime}')

# w_init_extra = np.zeros(X_train_extra.shape[1])
# w_grad_extra = linear_gradient_descent(X_train_extra, y_train_extra, w_init_extra, decay_speed=10**(-10), learn_rate=10**(-8), min_err=10**(-7), max_iter=10000000, verbose=True)
w_closed_extra = linear_closed_form(X_train_extra, y_train_extra)

# Compute MSE on training set
y_grad_train_extra = np.matmul(X_train_extra, w_closed_extra)
mse_grad_train_extra = np.sum((y_grad_train_extra - y_train_extra)**2)/len(y_train_extra)
print(f'MSE Training: {mse_grad_train_extra}')

# Compute MSE on validation set
y_grad_valid_extra = np.matmul(X_valid_extra, w_closed_extra)
mse_grad_valid_extra = np.sum((y_grad_valid_extra - y_valid_extra)**2)/len(y_valid_extra)
print(f'MSE Validation: {mse_grad_valid_extra}')

print(f'MSE Training Improvement: {mse_grad_train - mse_grad_train_extra}')
print(f'MSE Validation Improvement: {mse_grad_valid - mse_grad_valid_extra}')

# Compute features on test set
X_test, y_test = ppd.compute_features(test, extra_features=True, num_word_features=num_words)
print(X_test.shape)

# Compute MSE on test set
y_lin_test = np.matmul(X_test, w_closed_extra)
mse_test = np.sum((y_lin_test - y_test)**2)/len(y_test)
print(f'MSE Test: {mse_test}')
