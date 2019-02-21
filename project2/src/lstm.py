import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from sklearn.model_selection import train_test_split

from data import load_train

data = load_train()
reviews, Y = zip(*data)

max_features = 5000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(reviews)
X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X)
print(len(X))

embed_size = 32
lstm_size = 100
batch_size = 64

model = Sequential()
model.add(Embedding(max_features, embed_size, input_length=X.shape[1]))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(lstm_size))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

start = time.time()
history = model.fit(X_train, Y_train, epochs=7, batch_size=batch_size)
print(f'Training time: {time.time()-start}')

score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Accuracy: %.2f' % acc)

model.save('lstm.h5')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
