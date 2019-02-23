import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from data import load_train

data = load_train()
reviews, y = zip(*data)

max_features = 5000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(reviews)
X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X)
print(len(X))

embed_size = 64
lstm_size = 128
batch_size = 128
epochs = 3

model = Sequential()
model.add(Embedding(max_features, embed_size, input_length=X.shape[1]))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=lstm_size, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

start = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
print(f'Training time: {time.time()-start}')

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f'Accuracy: {acc:.2f}')

model.save('./project2/data/model/lstm.h5')

history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

print(f'Training accuracy: {acc}')
print(f'Validation accuracy: {val_acc}')
print(f'Training loss: {loss}')
print(f'Validation loss: {val_loss}')

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
