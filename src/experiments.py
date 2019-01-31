#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'src'))
	print(os.getcwd())
except:
	pass

#%%
import time
import numpy as np
import matplotlib.pyplot as plt

from features import PreprocessData
from linear_regression import linear_closed_form, linear_gradient_descent

ppd = PreprocessData()

# Split dataset
train, validation, test = ppd.preprocess_data(ppd.data)


#%%
# Compute most common words from 
ppd.compute_most_common_words(train)

X, y = ppd.compute_features(train)
print(X.shape)
print(y.shape)


#%%
w_closed = linear_closed_form(X, y)
print(w_closed.shape)
print(w_closed)


#%%
w_init = np.zeros(X.shape[1])
decay_speed = 10**(-12)
learn_rate = 4*10**(-8)
min_err = 10**(-7)
max_iter = 10000000

start = time.time()
w_grad = linear_gradient_descent(X, y, w_init, decay_speed, learn_rate, min_err, max_iter)
print(w_grad.shape)
print(w_grad)
print(time.time() - start)


#%%
y_closed = np.matmul(X, w_closed)
mse_closed = np.sum((y_closed - y)**2)/len(y)
print(mse_closed)

y_grad = np.matmul(X, w_grad)
mse_grad = np.sum((y_grad - y)**2)/len(y)
print(mse_grad)


#%%



